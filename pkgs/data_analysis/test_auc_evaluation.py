"""AUC regression checks for experiment scripts and the shared evaluator."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.data_analysis import auc_evaluation as auc
from pkgs.data_analysis import clinical_validity_analysis as cv
from pkgs.data_analysis.types import ExperimentScenario


class SharedAUCTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            'subject_id': [1, 2, 3, 4, 5],
            'duration_in_days': [100., 500., 1000., 2000., 3907.038194444444],
            'has_esrd': [1, 0, 1, 0, 1],
        })
        self.test = pd.DataFrame({
            'subject_id': [10, 11, 12, 13],
            'duration_in_days': [100., 365., 1000., 4333.0256944444445],
            'has_esrd': [1, 1, 0, 1],
        })
        self.risk = np.array([.9, .5, .1, .2])

    def test_late_event_uses_same_supported_auc_in_experiments_and_analysis(self):
        y_train = cv.build_censoring_reference(self.train)
        y_test = cv.build_censoring_reference(self.test)
        times = np.arange(100., 730.)
        with self.assertRaisesRegex(ValueError, 'largest observed time point'):
            cumulative_dynamic_auc(y_train, y_test, self.risk, times)

        cap, _ = cv.evaluation_time_cap(self.train)
        d, e, count = cv._administratively_censor(
            self.test.duration_in_days, self.test.has_esrd, cap)
        self.assertEqual(count, 1)
        self.assertFalse(e[-1])
        expected = cumulative_dynamic_auc(y_train, Surv.from_arrays(e, d), self.risk, times)[1]
        with contextlib.redirect_stdout(io.StringIO()) as log:
            actual = auc.report_auc(self.train, self.test, self.risk)
        self.assertAlmostEqual(actual, expected)
        self.assertIn('Mean time-dependent AUC (patient-level', log.getvalue())
        result = cv.discrimination_metrics(
            y_train, cap, self.risk, self.test.duration_in_days.to_numpy(),
            self.test.has_esrd.to_numpy(), None, None)
        self.assertEqual(result['auc'], round(actual, 4))
        self.assertNotIn('auc', result['errors'])
        self.assertEqual(self.test.has_esrd.iloc[-1], 1)
        self.assertGreater(self.test.duration_in_days.iloc[-1], cap)

    def test_lab_rows_are_reduced_with_their_scores_in_patient_order(self):
        early_train = self.train.assign(duration_in_days=0., has_esrd=0)
        early_test = self.test.assign(duration_in_days=0., has_esrd=0)
        train = pd.concat([early_train, self.train], ignore_index=True)
        test = pd.concat([early_test, self.test], ignore_index=True)
        scores = np.r_[np.array([100., -50., 80., 40.]), self.risk]
        order = np.array([7, 0, 5, 3, 6, 2, 4, 1])
        with contextlib.redirect_stdout(io.StringIO()):
            expected = auc.report_auc(self.train, self.test, self.risk)
            actual = auc.report_auc(train, test.iloc[order], scores[order])
        self.assertAlmostEqual(actual, expected)

    def test_flat_frames_and_patient_prediction_adapter_agree(self):
        model = SimpleNamespace(predictions=Mock(return_value=(
            self.risk, self.test.duration_in_days.to_numpy(),
            self.test.has_esrd.to_numpy(), None)))
        with contextlib.redirect_stdout(io.StringIO()):
            flat = auc.report_auc(self.train.drop(columns='subject_id'),
                                  self.test.drop(columns='subject_id'), self.risk)
            predicted = auc.report_prediction_auc(
                model, ExperimentScenario.FOUR_FEATURES, self.train, self.test)
        self.assertAlmostEqual(flat, predicted)
        model.predictions.assert_called_once_with(ExperimentScenario.FOUR_FEATURES)
        model.predictions.return_value = (self.risk, np.ones(4), np.ones(4), None)
        with self.assertRaisesRegex(ValueError, 'not the same outcome set|durations differ'):
            auc.report_prediction_auc(model, ExperimentScenario.FOUR_FEATURES, self.train, self.test)

    def test_undefined_auc_is_unavailable_without_fabricated_half_score(self):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            value = auc.report_auc(self.train, self.test, np.ones(4))
        self.assertIsNone(value)
        self.assertIn('numerically constant', log.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as log:
            value = auc.report_auc(self.train, self.test.assign(has_esrd=0), self.risk)
        self.assertIsNone(value)
        self.assertIn('No observed events', log.getvalue())

    def test_neural_prediction_adapter_restores_model_state(self):
        import torch

        class Predictor(torch.nn.Module):
            def __init__(self, predictions):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.result = predictions

            def predictions(self, scenario):
                assert self.weight.device.type == 'cpu'
                self.eval()
                return self.result

        model = Predictor((self.risk, self.test.duration_in_days.to_numpy(),
                           self.test.has_esrd.to_numpy(), None))
        with contextlib.redirect_stdout(io.StringIO()):
            result = auc.report_prediction_auc(
                model, ExperimentScenario.FOUR_FEATURES, self.train, self.test)
        self.assertTrue(np.isfinite(result))
        self.assertTrue(model.training)
        self.assertEqual(model.weight.device.type, 'cpu')

    def test_cox_trains_saves_and_reports_replacement_auc(self):
        from pkgs.experiments import cox

        rng = np.random.default_rng(23)
        def frame(durations, events):
            n = len(durations)
            return pd.DataFrame({
                'subject_id': np.arange(n), 'start': np.zeros(n), 'stop': durations,
                'duration_in_days': durations, 'has_esrd': events,
                'age': rng.uniform(40., 80., n), 'gender': np.arange(n) % 2,
                'egfr': rng.uniform(20., 60., n), 'uacr': rng.uniform(10., 200., n),
            })
        train = frame(np.arange(1., 31.) * 100., np.arange(30) % 3 != 0)
        test = frame([.5, 100., 200., 400., 600., 1200., 2000., 5000.],
                     [0, 1, 1, 0, 1, 0, 0, 1])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as log:
            artifact = Path(tmp) / 'cox.dill'
            with patch.object(cox, 'get_train_test_data', return_value=(train, test)), \
                    patch.object(cox, 'get_model_path', return_value=str(artifact)), \
                    patch.object(cox, 'report_auc', wraps=auc.report_auc) as report:
                cox.run_cox_model(ExperimentScenario.FOUR_FEATURES)
            self.assertTrue(artifact.is_file())
            report.assert_called_once()
        self.assertIn('Concordance Index Test:', log.getvalue())
        self.assertIn('Mean time-dependent AUC (patient-level', log.getvalue())
        self.assertNotIn('AUC unavailable', log.getvalue())


if __name__ == '__main__':
    unittest.main()
