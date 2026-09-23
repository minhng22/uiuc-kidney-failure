"""Check the default analysis command and gap-audit dispatch without model fitting."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from pkgs import commons
from pkgs.scripts import run_experiments as runner
from pkgs.scripts import audit_outcome_definition as outcome
from pkgs.scripts import audit_prediction_time as timing


class AnalysisRunnerTests(unittest.TestCase):
    def test_analysis_workers_train_missing_selected_models_then_reuse_them(self):
        from pkgs.data_analysis.types import ExperimentScenario

        analyzers = {
            'clinical_validity': ('clinical_validity_analysis', 'ClinicalValidityAnalyzer'),
            'feature_importance': ('feature_importance_analysis', 'FeatureImportanceAnalyzer'),
            'subgroup': ('subgroup_analysis', 'SubgroupAnalyzer'),
        }
        for task, (module_name, class_name) in analyzers.items():
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                for split in ('train', 'test'):
                    (output / f'four_features_{split}_data.csv').touch()
                existing = output / 'four_features_cox_model.dill'
                existing.write_text('existing model')
                ddh = output / 'four_features_ddh_model.pt'
                rnn = output / 'four_features_rnn_surv_model.pt'
                train_ddh = Mock(side_effect=lambda scenario: ddh.touch())
                train_rnn = Mock(side_effect=lambda scenario: rnn.touch())
                modules = {
                    'pkgs.experiments.dynamic_deephit': SimpleNamespace(run=train_ddh),
                    'pkgs.experiments.rnnsurv': SimpleNamespace(run=train_rnn),
                }
                analyzer = Mock(models=['cox', 'ddh', 'rnn_surv', 'srf'])
                analyzer.all_results = {}
                analyzer.all_importances = {}

                def analyze():
                    self.assertTrue(ddh.is_file())
                    self.assertTrue(rnn.is_file())
                    self.assertEqual(existing.read_text(), 'existing model')
                    analyzer.all_results['four_features'] = {'cox': {}}
                    analyzer.all_importances['four_features'] = {'cox': {}}

                analyzer.analyze_four_features.side_effect = analyze
                fake_module = SimpleNamespace(**{class_name: Mock(return_value=analyzer)})
                with patch.dict(runner.sys.modules, {
                    f'pkgs.data_analysis.{module_name}': fake_module,
                }), patch.object(commons, 'generate_data_path_latest_rep', tmp), \
                        patch.dict(runner.os.environ), \
                        patch.object(runner.importlib, 'import_module', side_effect=modules.__getitem__) as load, \
                        contextlib.redirect_stdout(io.StringIO()):
                    args = runner.parse_args([
                        'analyze', '--worker', '--reps', '99', '--analyses', task,
                        '--scenarios', 'four_features', '--models', 'cox', 'dynamic_deephit', 'rnnsurv',
                    ])
                    self.assertEqual(runner.run_worker(args), 0)
                    self.assertEqual(runner.run_worker(args), 0)
                train_ddh.assert_called_once_with(ExperimentScenario.FOUR_FEATURES)
                train_rnn.assert_called_once_with(ExperimentScenario.FOUR_FEATURES)
                self.assertEqual(load.call_count, 2)
                self.assertEqual(analyzer.analyze_four_features.call_count, 2)

    def test_missing_model_training_failure_fails_worker(self):
        for error in (RuntimeError('training failed'), None):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                for split in ('train', 'test'):
                    Path(tmp, f'four_features_{split}_data.csv').touch()
                analyzer = Mock(models=['cox'], all_results={})
                fake_module = SimpleNamespace(ClinicalValidityAnalyzer=Mock(return_value=analyzer))
                train = Mock(side_effect=error)
                with patch.dict(runner.sys.modules, {
                    'pkgs.data_analysis.clinical_validity_analysis': fake_module,
                }), patch.object(commons, 'generate_data_path_latest_rep', tmp), \
                        patch.dict(runner.os.environ), \
                        patch.object(runner.importlib, 'import_module',
                                     return_value=SimpleNamespace(run_cox_model=train)), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()) as errors:
                    args = runner.parse_args([
                        'analyze', '--worker', '--reps', '99', '--analyses', 'clinical_validity',
                        '--scenarios', 'four_features', '--models', 'cox',
                    ])
                    self.assertEqual(runner.run_worker(args), 1)
                train.assert_called_once()
                analyzer.analyze_four_features.assert_not_called()
                self.assertIn('training failed' if error else 'did not produce required artifact',
                              errors.getvalue())

    def test_artifact_paths_match_training_paths(self):
        names = {'dynamic_deephit': 'dynamic_deep_hit', 'rnnsurv': 'rnn_surv'}
        self.assertEqual(set(runner.MODEL_ARTIFACT_SUFFIXES), set(runner.TRAIN_FUNCTIONS))
        for scenario in runner.SCENARIOS:
            for model, suffix in runner.MODEL_ARTIFACT_SUFFIXES.items():
                if model == 'kfre':
                    continue
                with self.subTest(scenario=scenario, model=model):
                    actual = getattr(commons, f'{scenario}_{names.get(model, model)}_model_path')
                    self.assertEqual(Path(actual), Path(commons.generate_data_path_latest_rep)
                                     / f'{scenario}_{suffix}')

    def test_kfre_cache_generated_only_for_supported_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            train = Mock(side_effect=lambda scenario:
                         (output / f'{scenario.value}_kfre_2yr_risk_scores.csv').touch())
            with patch.object(runner.importlib, 'import_module',
                              return_value=SimpleNamespace(run_kfre_model=train)), \
                    contextlib.redirect_stdout(io.StringIO()):
                for scenario in runner.SCENARIOS:
                    self.assertEqual(runner.ensure_analysis_models(output, scenario, ['kfre']), [])
            self.assertEqual([call.args[0].value for call in train.call_args_list],
                             ['four_features', 'eight_features'])

    def test_production_directory_names_in_data_builder_and_runner(self):
        from pkgs.paths import repetition_directory_name
        from pkgs.scripts.build_external_validation_reps import rep_dir
        for rep in range(1, 6):
            self.assertEqual(repetition_directory_name(rep), f'rep_{rep}')
            self.assertEqual(Path(rep_dir(rep)).name, f'rep_{rep}')
        self.assertEqual(repetition_directory_name(99), 'rep99')
        self.assertEqual(repetition_directory_name(100), 'rep100')
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(runner.main(['analyze', '--reps', 'all', '--dry-run']), 0)
        for rep in range(1, 6):
            self.assertIn(f'generated_data/rep_{rep}/', out.getvalue())

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
