"""Regression checks for the five evaluation-review findings; no model files needed.

Run: python -m unittest pkgs.data_analysis.test_evaluation_gaps
"""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from pkgs.data_analysis import bootstrap_ci as bc
from pkgs.data_analysis import clinical_validity_analysis as cv
from pkgs.data_analysis.patient_outcomes import patient_level_egfr, patient_level_outcomes
from pkgs.data_analysis.subgroup_analysis import SubgroupAnalyzer
from pkgs.scripts import aggregate_rep_metrics as aggregate


class BootstrapOutcomeTests(unittest.TestCase):
    def test_c_index_keeps_late_comparable_pairs_while_ipcw_uses_cap(self):
        d, e, r = np.array([1., 2., 3., 4.]), np.ones(4, bool), np.array([4., 3., 1., 2.])
        dc, ec, _ = cv._administratively_censor(d, e, 2.)
        y = Surv.from_arrays([True, False, True, False], [1., 2., 3., 5.])
        times = np.array([1., 1.5])
        survival = np.full((4, 2), .5)
        result = bc.bootstrap_model_metrics(
            y, d, e, r, survival, times, times, np.arange(4)[None, :],
            ipcw_durations=dc, ipcw_events=ec)
        self.assertAlmostEqual(result['c_index'][0], 5 / 6)
        self.assertNotAlmostEqual(result['c_index'][0], concordance_index(dc, -r, ec))
        capped = Surv.from_arrays(ec, dc)
        self.assertAlmostEqual(result['brier'][0], integrated_brier_score(y, capped, survival, times))
        self.assertAlmostEqual(result['auc'][0], cumulative_dynamic_auc(y, capped, r, times)[1])

    def test_cohort_and_subgroup_intervals_use_original_outcomes(self):
        d = np.arange(20, dtype=float) * 100 + 1
        e = np.ones(20, bool)
        r = np.arange(20, dtype=float) % 7
        terminal = pd.DataFrame({'subject_id': np.arange(20), 'duration_in_days': d, 'has_esrd': e})
        y = Surv.from_arrays(e, d)
        expected = np.array([concordance_index(d[ix], -r[ix], e[ix])
                             for ix in bc.bootstrap_indices(20, 100)])
        native = lambda t: np.full(20, .5)
        with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
            analyzer = cv.ClinicalValidityAnalyzer(out)
            point = cv.discrimination_metrics(y, 800., r, d, e, None, native)
            with patch.dict(os.environ, {'CKD_N_BOOTSTRAP': '100'}):
                result = analyzer.run_bootstrap('test', y, 800., terminal,
                    {'cox': (r, d, e, native)}, {'cox': None}, {'cox': point})
            self.assertEqual(result['summaries']['cox']['c_index'], bc.summarize(point['c_index'], expected))
            subgroup = SubgroupAnalyzer(out)._group_metrics(
                y, 800., np.arange(20), d, e, r, None, native, 100)
            self.assertEqual(subgroup['c_index'], bc.summarize(point['c_index'], expected))


class CensoringSupportTests(unittest.TestCase):
    def test_event_censor_ties_use_sksurv_support(self):
        terminal = pd.DataFrame({'duration_in_days': [1., 2., 2.], 'has_esrd': [1, 1, 0]})
        y = cv.build_censoring_reference(terminal)
        cap, maximum = cv.evaluation_time_cap(terminal)
        self.assertEqual(maximum, 2.)
        self.assertLess(cap, 2.)
        self.assertGreater(cap, 1.9)
        censor = CensoringDistributionEstimator().fit(y)
        self.assertGreater(censor.predict_proba([cap])[0], 0)
        self.assertEqual(censor.predict_proba([2.])[0], 0)
        d, e, _ = cv._administratively_censor([1., 2., 3.], [1, 1, 0], cap)
        test = Surv.from_arrays(e, d)
        _, auc = cumulative_dynamic_auc(y, test, [3., 2., 1.], [1., 1.5])
        self.assertTrue(np.isfinite(auc))
        self.assertTrue(np.isfinite(integrated_brier_score(y, test, np.full((3, 2), .5), [1., 1.5])))

    def test_positive_terminal_censoring_survival_keeps_maximum(self):
        terminal = pd.DataFrame({'duration_in_days': [1., 2., 3.], 'has_esrd': [1, 0, 1]})
        self.assertEqual(cv.evaluation_time_cap(terminal), (3., 3.))

    def test_no_nonnegative_support_fails_explicitly(self):
        terminal = pd.DataFrame({'duration_in_days': [0., 0.], 'has_esrd': [1, 0]})
        with self.assertRaisesRegex(ValueError, 'no supported nonnegative time'):
            cv.evaluation_time_cap(terminal)


class NumericalRankingTests(unittest.TestCase):
    def test_aggregation_keeps_brier_when_rank_metrics_are_withheld(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rep, row in [(1, 'None brier=0.4 (native) auc=None'),
                             (2, '0.6 brier=0.2 auc=0.7')]:
                path = root / 'generated_data' / f'rep{rep}' / 'test_clinical_validity_report.txt'
                path.parent.mkdir(parents=True)
                path.write_text('Discrimination metrics:\n  Cox: c_index=' + row + '\n\n')
            parsed = aggregate.parse_report(root / 'generated_data/rep1/test_clinical_validity_report.txt')
            self.assertEqual(parsed, {'Cox': (None, .4, None)})
            output = io.StringIO()
            with patch.object(aggregate, 'REPO_ROOT', root), \
                 patch.object(aggregate, 'REP_RANGE', range(1, 3)), contextlib.redirect_stdout(output):
                aggregate.aggregate_scenario('test')
            self.assertIn('C-index=[2]; Brier=[1, 2]; AUC=[2]', output.getvalue())

    def test_float32_roundoff_withholds_ranks_but_keeps_native_brier(self):
        r = np.array([.5, .50000006, .50000012, .5, .50000006, .5], dtype=np.float32)
        d, e = np.arange(6, dtype=float), np.array([1, 1, 0, 1, 0, 0], bool)
        y = Surv.from_arrays([True, True, True, False, True, False], [0., 1., 2., 3., 4., 6.])
        native = lambda t: np.full(6, .4)
        point = cv.discrimination_metrics(y, 5.9, r, d, e, None, native, auc_horizon_days=3)
        self.assertIsNone(point['c_index'])
        self.assertIsNone(point['auc'])
        self.assertIn('numerically constant', point['errors']['auc'])
        self.assertTrue(np.isfinite(point['brier']))
        reps = bc.bootstrap_model_metrics(y, d, e, r, np.full((6, 2), .6),
            np.array([1., 2.]), np.array([1., 2.]), bc.bootstrap_indices(6, 100))
        self.assertTrue(np.isnan(reps['c_index']).all())
        self.assertTrue(np.isnan(reps['auc']).all())
        self.assertGreater(np.isfinite(reps['brier']).sum(), 20)
        self.assertIsNone(bc.paired_difference(reps['auc'], np.ones(100)))

    def test_guard_uses_original_precision_and_scale(self):
        self.assertIsNone(bc.discrimination_unavailable_reason(np.array([.5, .50000012], dtype=np.float64)))
        self.assertIsNone(bc.discrimination_unavailable_reason(np.array([1e-20, 2e-20], dtype=np.float32)))
        self.assertIsNotNone(bc.discrimination_unavailable_reason(np.zeros(4)))
        self.assertIsNotNone(bc.discrimination_unavailable_reason([1., np.inf]))


class DecisionCohortTests(unittest.TestCase):
    def setUp(self):
        self.d, self.e = np.array([1., 1., 1., 10., 10., 10.]), np.array([1, 1, 1, 0, 0, 0])

    def test_missing_predictions_restrict_every_strategy(self):
        result = cv.decision_curve_comparison(
            {'cox': [np.nan] * 3 + [1.] * 3, 'ddh': np.ones(6)},
            self.d, self.e, 2, np.full(6, 20.), thresholds=[.2])
        self.assertEqual(result['n_included'], 3)
        self.assertEqual(result['treat_all'][.2], -.25)
        self.assertEqual(result['models']['cox'][.2], -.25)
        self.assertEqual(result['models']['ddh'][.2], -.25)
        self.assertEqual(result['egfr_nb'][30]['net_benefit'][.2], -.25)

    def test_missing_egfr_restricts_models_and_treat_all(self):
        result = cv.decision_curve_comparison({'cox': np.ones(6)}, self.d, self.e, 2,
            [np.nan, np.inf, np.nan, 20., 20., 20.], thresholds=[.2])
        self.assertEqual(result['n_included'], 3)
        self.assertEqual(result['models']['cox'], result['treat_all'])
        self.assertEqual(result['treat_all'][.2], -.25)

    def test_unavailable_strategies_are_omitted_without_emptying_cohort(self):
        result = cv.decision_curve_comparison({'cox': np.ones(6), 'ddh': np.full(6, np.nan)},
            self.d, self.e, 2, np.full(6, np.nan), thresholds=[.2])
        self.assertEqual(result['n_included'], 6)
        self.assertEqual(result['unavailable_models'], ['ddh'])
        self.assertIsNone(result['egfr_nb'])
        self.assertEqual(result['treat_all'][.2], .375)

    def test_disjoint_availability_does_not_compare_different_cohorts(self):
        result = cv.decision_curve_comparison(
            {'cox': [1.] * 3 + [np.nan] * 3, 'ddh': [np.nan] * 3 + [1.] * 3},
            self.d, self.e, 2, thresholds=[.2])
        self.assertEqual(result['n_included'], 0)
        self.assertIsNone(result['treat_all'][.2])
        self.assertIsNone(result['models']['cox'][.2])

    def test_egfr_uses_last_finite_measured_value_before_landmark(self):
        raw = pd.DataFrame({'subject_id': [1, 1, 1, 2, 3, 3, 4],
            'duration_in_days': [0., 1., 3., 1., 0., 1., 1.],
            'has_esrd': [0, 0, 0, 0, 0, 0, 0],
            'egfr': [25., np.nan, 10., np.nan, 0., np.inf, 15.],
            'egfr_missing': [0, 0, 0, 0, 1, 0, 0]})
        terminal = patient_level_outcomes(raw)
        terminal.loc[terminal.subject_id == 1, 'duration_in_days'] = 1.
        selected, info = patient_level_egfr(raw, terminal)
        self.assertEqual(selected.subject_id.tolist(), [1, 4])
        self.assertEqual(selected.egfr.tolist(), [25., 15.])
        self.assertEqual(info['patients_without_egfr'], 2)
        self.assertEqual(info['rows_with_measured_egfr'], 2)

    def test_analyzer_reports_and_plots_the_common_cohort(self):
        # Exercise the real orchestration and subject-ID join, not just the
        # curve helper: prediction and eGFR coverage differ in this cohort.
        raw = pd.DataFrame({'subject_id': range(6), 'duration_in_days': self.d,
                            'has_esrd': self.e, 'egfr': [np.nan] * 3 + [20.] * 3})
        r = np.arange(6, dtype=float)
        native = lambda t: np.ones(6)
        with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
            analyzer = cv.ClinicalValidityAnalyzer(out)
            with patch.object(analyzer, '_model_paths', return_value={'cox': __file__}), \
                 patch.object(analyzer, '_get_predictions', return_value=(r, self.d, self.e, native)), \
                 patch.object(analyzer, '_get_train_risk_scores', return_value=(r, self.d, self.e)), \
                 patch.object(analyzer, 'run_bootstrap'), \
                 patch.object(analyzer, 'create_calibration_plot'), \
                 patch.object(analyzer, 'create_decision_curve_plot'), \
                 patch.object(analyzer, 'save_scenario_report'):
                analyzer.analyze_scenario('synthetic', None, raw, raw)
            result = analyzer.all_results['synthetic']['horizons'][5.]
            self.assertEqual(result['dca_subject_ids'], [3, 4, 5])
            self.assertEqual(result['dca_n_patients'], 3)
            self.assertEqual(result['models']['cox']['model_nb'], result['treat_all'])
            self.assertEqual(result['treat_all'][.2], -.25)


if __name__ == '__main__':
    unittest.main()
