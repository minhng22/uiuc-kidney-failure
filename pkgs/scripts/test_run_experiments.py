"""Check the default analysis command and gap-audit dispatch without model fitting."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from pkgs import commons
from pkgs.scripts import run_experiments as runner
from pkgs.scripts import audit_outcome_definition as outcome
from pkgs.scripts import audit_prediction_time as timing


class AnalysisRunnerTests(unittest.TestCase):
    def test_parallel_defaults_launch_all_five_tasks_for_all_five_reps(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            status = runner.main(['analyze', '--reps', 'all', '--parallel-reps', '--log-dir', tmp])
        self.assertEqual(status, 0)
        commands = [call.args[0] for call in run.call_args_list]
        tasks = {(int(cmd[cmd.index('--reps') + 1]), cmd[cmd.index('--analyses') + 1])
                 for cmd in commands}
        self.assertEqual(tasks, {(rep, task) for rep in range(1, 6) for task in (
            'clinical_validity', 'feature_importance', 'subgroup', 'outcome_definition', 'prediction_time')})
        self.assertEqual(len(commands), 25)
        for call in run.call_args_list:
            args = runner.parse_args(call.args[0][4:])
            self.assertTrue(args.worker)
            self.assertEqual(args.scenarios, list(runner.SCENARIOS))
            self.assertEqual(call.kwargs['env']['CKD_REP'], str(args.reps[0]))

    def test_explicit_selection_and_failure_propagation(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner.subprocess, 'run', side_effect=[
                    SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            status = runner.main(['analyze', '--reps', '99', '--analyses',
                                  'outcome_definition', 'prediction_time', '--log-dir', tmp])
        self.assertEqual(status, 1)
        self.assertEqual(run.call_count, 2)

    def test_audit_workers_dispatch_selected_scenarios_and_full_timing(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(commons, 'generate_data_path_latest_rep', tmp), \
                patch.dict(runner.os.environ), contextlib.redirect_stdout(io.StringIO()):
            for split in ('train', 'test'):
                Path(tmp, f'four_features_{split}_data.csv').touch()
            for task, module in [('outcome_definition', outcome), ('prediction_time', timing)]:
                with self.subTest(task=task), patch.object(module, 'main', return_value=0) as main:
                    args = runner.parse_args(['analyze', '--reps', '99', '--worker',
                                              '--analyses', task, '--scenarios', 'four_features'])
                    self.assertEqual(runner.run_worker(args), 0)
                    expected = ['--scenarios', 'four_features']
                    if task == 'prediction_time':
                        expected.append('--lab-timing')
                    main.assert_called_once_with(expected)
            with patch.object(timing, 'main') as main:
                args.scenarios = ['eight_features']
                self.assertEqual(runner.run_worker(args), 1)
                main.assert_not_called()

    def test_outcome_audit_honors_scenario_selection(self):
        diagnoses = pd.DataFrame({'subject_id': [1], 'hadm_id': [2], 'icd_code': ['N19']})
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(commons, 'generate_data_path_latest_rep', tmp), \
                patch.object(outcome.pd, 'read_csv', return_value=diagnoses), \
                patch.object(outcome, 'audit_codes'), patch.object(outcome, 'audit_cohort'), \
                patch.object(outcome, 'audit_horizon') as horizon, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(outcome.main(['--scenarios', 'eight_features']), 0)
            self.assertEqual(horizon.call_args.args[1], ['eight_features'])
            self.assertTrue(Path(tmp, 'stage_gap6_outcome_definition_report.txt').is_file())

    def test_timing_cache_restricts_union_to_scenario_without_rescanning(self):
        labs = pd.DataFrame({'subject_id': [1, 2, 3], 'itemid': ['51070'] * 3})
        with patch.dict(timing._LABS_CACHE, {frozenset({1, 2, 3}): labs}, clear=True), \
                patch.object(timing.pd, 'read_csv') as read:
            subset, _ = timing._load_cohort_labs([], {2})
            self.assertEqual(subset.subject_id.tolist(), [2])
            read.assert_not_called()

    def test_zero_duration_is_not_reported_as_single_row(self):
        df = pd.DataFrame({'subject_id': [1, 1, 2], 'duration_in_days': [0., 0., 0.],
                           'has_esrd': [1, 1, 0]})
        lines = []
        with contextlib.redirect_stdout(io.StringIO()):
            timing.audit_exported(lines, 'test', df, df)
        report = '\n'.join(lines)
        self.assertIn('zero follow-up duration: 2/2', report)
        self.assertIn('single exported row: 1/2', report)

    def test_full_timing_audit_scans_once_and_writes_both_scenarios(self):
        from pkgs.data_analysis import model_data_store

        pairs = {
            name: (pd.DataFrame({'subject_id': [sid], 'duration_in_days': [0.], 'has_esrd': [1]}),
                   pd.DataFrame({'subject_id': [sid], 'duration_in_days': [0.], 'has_esrd': [1]}))
            for name, sid in [('four_features', 1), ('eight_features', 2)]
        }
        raw = pd.DataFrame({
            'subject_id': [1, 1, 2, 2], 'hadm_id': [10, 10, 20, 20],
            'itemid': ['50912', '51070', '50912', '51070'],
            'charttime': ['2020-01-10', '2020-01-09'] * 2, 'valuenum': [1., 20., 1., 30.],
        })
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp, 'labs.csv')
            raw.to_csv(raw_path, index=False)
            for name in pairs:
                for split in ('train', 'test'):
                    Path(tmp, f'{name}_{split}_data.csv').touch()
            with patch.object(commons, 'generate_data_path_latest_rep', tmp), \
                    patch.object(commons, 'lab_events_file_path', str(raw_path)), \
                    patch.object(model_data_store, 'get_train_test_data',
                                 side_effect=lambda scenario: pairs[scenario.value]), \
                    patch.dict(timing._LABS_CACHE, {}, clear=True), \
                    patch.object(timing.pd, 'read_csv', wraps=pd.read_csv) as read, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(timing.main(['--scenarios', *pairs, '--lab-timing']), 0)
                self.assertEqual(read.call_count, 1)
            report = Path(tmp, 'stage_gap8_prediction_time_audit_report.txt').read_text()
            self.assertIn('DRAW — four_features', report)
            self.assertIn('DRAW — eight_features', report)
            self.assertEqual(report.count('recorded AFTER the anchor creatinine: 0/1'), 2)


if __name__ == '__main__':
    unittest.main()
