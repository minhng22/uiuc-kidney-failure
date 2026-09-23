"""Prediction-time uACR matching checks; no raw data or model files required."""
import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd

from pkgs.data_analysis import time_series_store as ts
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.scripts import audit_prediction_time as audit


class UacrMatchingTests(unittest.TestCase):
    def setUp(self):
        self.anchor = pd.DataFrame({
            'subject_id': [1, 2, 3], 'hadm_id': [10, 20, 30],
            'charttime': pd.to_datetime(['2020-01-10'] * 3),
            'anchor_age': [60] * 3, 'gender': ['F'] * 3, 'egfr': [40.] * 3,
        })
        self.uacr = pd.DataFrame({
            'subject_id': [1, 1, 1, 2, 3], 'hadm_id': [8, 9, 10, 20, 30],
            'charttime': pd.to_datetime([
                '2018-01-01', '2019-01-01', '2020-01-11', '2020-01-10', '2020-01-11']),
            'uacr': [10., 20., 100., 30., 200.],
        })

    def test_latest_prior_across_admissions_exact_match_and_future_only(self):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            result = ts.merge_nearest_within_admission(
                self.anchor, self.uacr, 'uacr', by='subject_id', direction='backward')
        result = result.set_index('subject_id')
        self.assertEqual(result.loc[1, 'uacr'], 20.)
        self.assertEqual(result.loc[2, 'uacr'], 30.)
        self.assertTrue(pd.isna(result.loc[3, 'uacr']))
        matched = result.dropna(subset=['uacr'])
        self.assertTrue((matched.uacr_charttime <= matched.charttime).all())
        self.assertIn('|after_anchor=0|', log.getvalue())

    def test_empty_source_keeps_anchors_unmatched(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = ts.merge_nearest_within_admission(
                self.anchor, self.uacr.iloc[:0], 'uacr', by='subject_id', direction='backward')
        self.assertEqual(len(result), len(self.anchor))
        self.assertTrue(result.uacr.isna().all())

    def test_both_scenario_callers_drop_future_only_uacr(self):
        for scenario in (ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES):
            with self.subTest(scenario=scenario), contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(ts, 'get_egfr_df', return_value=self.anchor.copy()))
                stack.enter_context(patch.object(ts, 'get_uacr_df', return_value=self.uacr.copy()))
                for lab in ('calcium', 'phosphate', 'bicarbonate', 'serum_albumin'):
                    chem = self.anchor[['subject_id', 'hadm_id', 'charttime']].copy()
                    # Chemistry retains its existing nearest-within-24h behavior.
                    chem['charttime'] += pd.Timedelta(hours=1)
                    chem[lab] = 1.
                    stack.enter_context(patch.object(ts, f'get_{lab}_df', return_value=chem))
                result = ts.get_lab_df_for_scenario_name(self.anchor[['subject_id']], scenario)
                self.assertEqual(result.set_index('subject_id').uacr.to_dict(), {1: 20., 2: 30.})
                self.assertTrue((result.uacr_charttime <= result.time).all())

    def test_timing_audit_uses_backward_uacr_rule(self):
        anchors = self.anchor.assign(itemid=1, valuenum=1.)
        uacr = self.uacr.assign(itemid=2, valuenum=self.uacr.uacr)
        labs = pd.concat([anchors, uacr], ignore_index=True)[
            ['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum']]
        lines = []
        with patch.object(audit, '_load_cohort_labs', return_value=(
                labs, {'creatinine': {1}, 'uacr': {2}})), \
                contextlib.redirect_stdout(io.StringIO()):
            audit.audit_lab_timing(lines, 'four_features', self.anchor, self.anchor)
        report = '\n'.join(lines)
        self.assertIn('direction=backward', report)
        self.assertIn('recorded AFTER the anchor creatinine: 0/2', report)
        self.assertIn('matched 2/3 anchor rows', report)


if __name__ == '__main__':
    unittest.main()
