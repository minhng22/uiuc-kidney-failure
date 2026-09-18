"""Subgroup performance analysis — discrimination and survival-probability
metrics within age, sex and race groups (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 12).

How to run
----------
From the repository root, with the project Python environment active::

    # Subgroup performance for rep99, all three scenarios and applicable models:
    python -m pkgs.scripts.run_experiments analyze --reps 99 --analyses subgroup

    # Production reps, selected scenarios/models:
    python -m pkgs.scripts.run_experiments analyze --reps all --analyses subgroup \\
        --scenarios four_features eight_features --models cox kfre

Outputs `<scenario>_subgroup_performance_report.txt` under
generated_data/rep<N>/.

What this is and is not
-----------------------
Gap 12 records that the repo ALREADY has demographic description
(demographics.py's age/gender/race distributions, cohort_flow_analysis.py's
age/sex summaries for source and extracted cohorts) and that the blanket claim
"no demographic analysis was performed" was too broad. What was missing is
performance WITHIN those groups: cohort composition, including composition split
by ESRD status, says nothing about whether a model discriminates equally well
for the groups it contains.

This module fills that in and nothing more. It reports per-group patient and
event counts, event proportions, C-index, integrated Brier score and mean
time-dependent AUC, each with a bootstrap percentile interval, for every model
in the comparison. That is subgroup performance analysis. It is NOT a fairness
evaluation: it fixes no allocation, defines no fairness criterion, and a group
difference here has many possible causes (sample size, event rate, follow-up
length, measurement density, coding practice) that these numbers do not
distinguish between. The report says so in its own header rather than leaving
it to be inferred.

Design choices carried in from the rest of the evaluation
---------------------------------------------------------
- Patients, not lab rows: every group is a set of PATIENTS with one terminal
  outcome each (patient_outcomes.patient_level_outcomes()), matching Gap 5a.
- The IPCW censoring reference is the FULL training cohort's, not a per-group
  one. G(t) is a nuisance distribution, and re-estimating it inside a group of
  30 patients would add more noise than it removes bias. The consequence —
  that a group whose censoring pattern differs from the training cohort's is
  weighted by the cohort's pattern — is stated in the report.
- Survival probabilities prefer each model's native curve, falling back to the
  labelled fitted conversion, exactly as in clinical_validity_analysis.py
  (Gap 5b). The source is reported per model.
- Metrics are only computed for groups that can support them
  (MIN_GROUP_PATIENTS / MIN_GROUP_EVENTS); smaller groups still get their
  counts printed, so they are visible as present-but-unmeasurable rather than
  absent.
"""
import os
from datetime import datetime

import numpy as np

from pkgs.commons import current_rep
from pkgs.data_analysis import bootstrap_ci
from pkgs.data_analysis.clinical_validity_analysis import (
    ClinicalValidityAnalyzer, build_censoring_reference, evaluation_time_cap,
    fit_breslow_baseline_hazard, survival_prob_matrix, _administratively_censor,
    discrimination_metrics,
)
from pkgs.data_analysis.patient_metadata import patient_metadata, subgroup_definitions
from pkgs.data_analysis.patient_outcomes import (
    patient_level_outcomes, verify_patient_outcomes, align_predictions_to_patients,
)

# A C-index needs enough comparable (event, later-survivor) pairs to mean
# anything, and an IPCW-weighted Brier/AUC needs enough events inside the
# evaluation window. These thresholds are deliberately conservative: below them
# the report prints the group's counts and says the metrics were not computed,
# rather than printing a number whose interval would span most of [0, 1].
MIN_GROUP_PATIENTS = 20
MIN_GROUP_EVENTS = 5
EVAL_HORIZON_DAYS = 730


class SubgroupAnalyzer(ClinicalValidityAnalyzer):
    """Reuses ClinicalValidityAnalyzer's model dispatch (_model_paths /
    _get_predictions / _get_train_risk_scores) and logging, and replaces its
    analyze_scenario with a per-subgroup one. Subclassing rather than copying
    keeps the two analyses reading the same model artifacts through the same
    code path, so a subgroup number and an overall number for the same model
    can never diverge because one of them loaded the model differently."""

    def analyze_scenario(self, scenario_name, scenario_enum, df_train, df_test):
        self.current_scenario = scenario_name
        self.log("=" * 80)
        self.log(f"SUBGROUP PERFORMANCE ANALYSIS - {scenario_name.upper()}")
        self.log("=" * 80)
        self.log("Subgroup PERFORMANCE, not a fairness evaluation — see this module's docstring.")
        self.log("")

        train_terminal = patient_level_outcomes(df_train)
        test_terminal = patient_level_outcomes(df_test)
        train_check = verify_patient_outcomes(df_train, train_terminal)
        test_check = verify_patient_outcomes(df_test, test_terminal)
        for split_name, check in (('train', train_check), ('test', test_check)):
            self.log(f"{split_name}: {check['n_rows']} lab-event rows -> {check['n_patients']} patients, "
                     f"{check['n_events']} events (patient-level event rate "
                     f"{check['patient_event_rate']:.4f})")
            for problem in check['problems']:
                self.log(f"  WARNING {split_name}: {problem}")

        y_train = build_censoring_reference(train_terminal)
        train_max, train_observed_max = evaluation_time_cap(train_terminal)
        self.log(f"IPCW censoring reference: the full training cohort "
                 f"({train_check['n_patients']} patients), "
                 f"not a per-group one; evaluation capped at {train_max:.1f} days "
                 f"(training follow-up {train_observed_max:.1f} days).")

        metadata, metadata_info = patient_metadata(test_terminal)
        self.log("")
        self.log("Patient metadata join:")
        self.log(f"  {metadata_info['patients']} held-out patients; "
                 f"{metadata_info['patients_missing_from_patients_csv']} with no patients.csv row")
        self.log(f"  Age variable — {metadata_info['age_variable']}")
        offsets = metadata_info['landmark_offset_years']
        self.log(f"  Time from each patient's first lab record to their prediction landmark: "
                 f"median {offsets['median']} y, p90 {offsets['p90']} y, max {offsets['max']} y — "
                 "the landmark can sit this far from the anchor year the age is measured at.")
        race_info = metadata_info['race']
        self.log(f"  Race — {race_info['selection_rule']}")
        self.log(f"    {race_info['patients_with_recorded_race']}/{race_info['patients']} patients have a "
                 f"recorded race; {race_info['patients_without_recorded_race']} retained as unknown; "
                 f"{race_info['patients_with_more_than_one_recorded_race']} have more than one recorded "
                 f"value across admissions; {race_info['admission_rows_with_unrecorded_race']} admission "
                 "rows carry declined/unobtainable/unknown.")

        durations = test_terminal['duration_in_days'].values.astype(float)
        events = test_terminal['has_esrd'].values.astype(int)

        predictions, baselines = self._load_predictions(
            scenario_name, scenario_enum, test_terminal, durations, events)
        if not predictions:
            self.log("\nNo usable model predictions for this scenario; nothing to stratify.")
            self.all_results[scenario_name] = {'groups': {}, 'metadata': metadata_info}
            self.save_scenario_report(scenario_name)
            return

        self.log("\nGroup composition (patients / events / event proportion):")
        group_results = {}
        for column, label in subgroup_definitions():
            self.log(f"\n--- {label} ---")
            group_results[column] = {}
            for value, group in metadata.groupby(column, observed=True):
                positions = np.asarray(group.index, dtype=int)
                n, n_events = len(positions), int(group['has_esrd'].sum())
                self.log(f"  {str(value):<24} n={n:<5} events={n_events:<5} "
                         f"event_rate={n_events / n:.4f}"
                         + ("" if n >= MIN_GROUP_PATIENTS and n_events >= MIN_GROUP_EVENTS
                            else f"   [metrics not computed: needs >= {MIN_GROUP_PATIENTS} patients "
                                 f"and >= {MIN_GROUP_EVENTS} events]"))
                group_results[column][str(value)] = {
                    'n': n, 'n_events': n_events, 'event_rate': round(n_events / n, 4),
                    'measurable': bool(n >= MIN_GROUP_PATIENTS and n_events >= MIN_GROUP_EVENTS),
                    'models': {},
                }

        n_bootstrap = int(os.environ.get('CKD_N_BOOTSTRAP', bootstrap_ci.DEFAULT_N_BOOTSTRAP))
        self.log(f"\nPer-group performance, {n_bootstrap} within-group bootstrap resamples, "
                 "95% percentile intervals:")
        self.log("  C-index and mean time-dependent AUC: higher is better. Integrated Brier "
                 f"(0-{EVAL_HORIZON_DAYS}d): lower is better.")

        for column, label in subgroup_definitions():
            self.log(f"\n=== {label} ===")
            for value, group in metadata.groupby(column, observed=True):
                entry = group_results[column][str(value)]
                if not entry['measurable']:
                    continue
                positions = np.asarray(group.index, dtype=int)
                self.log(f"\n  [{value}] n={entry['n']} events={entry['n_events']}")
                for model_name, (risk_scores, _, _, native_prob_fn) in predictions.items():
                    result = self._group_metrics(
                        y_train, train_max, positions, durations, events, risk_scores,
                        baselines.get(model_name), native_prob_fn, n_bootstrap)
                    entry['models'][model_name] = result
                    self.log(f"    {self.model_pretty_names[model_name]:<20} "
                             f"c_index={self._fmt(result['c_index'])}  "
                             f"brier={self._fmt(result['brier'])} [{result['brier_source']}]  "
                             f"auc={self._fmt(result['auc'])}")

        self.all_results[scenario_name] = {'groups': group_results, 'metadata': metadata_info}
        self.save_scenario_report(scenario_name)

    @staticmethod
    def _fmt(summary):
        if summary is None or summary.get('estimate') is None:
            return "n/a"
        if summary.get('ci_low') is None:
            return f"{summary['estimate']} [CI unavailable]"
        return f"{summary['estimate']} [{summary['ci_low']}, {summary['ci_high']}]"

    def _load_predictions(self, scenario_name, scenario_enum, test_terminal, durations, events):
        """Same load/align/baseline-fit sequence clinical_validity_analysis.py
        performs, so both analyses see identical predictions for a model. Every
        model is scored against the canonical patient-level terminal outcomes
        (Gap 5a); a model whose prediction row set does not match them is
        dropped rather than stratified against the wrong patients."""
        predictions, baselines = {}, {}
        for model_name, model_path in self._model_paths(scenario_name).items():
            if not os.path.exists(model_path):
                self.log(f"Model file not found, skipping: {model_path}")
                continue
            try:
                preds = self._get_predictions(model_name, model_path, scenario_enum)
                if preds is None:
                    self.log(f"No usable model at {model_path}, skipping {model_name}.")
                    continue
            except Exception as e:
                self.log(f"Error getting predictions for {model_name}: {e}")
                continue
            ok, message = align_predictions_to_patients(
                test_terminal, preds[1], preds[2], self.model_pretty_names[model_name])
            if not ok:
                self.log(f"  EXCLUDED — {message}")
                continue
            predictions[model_name] = (preds[0], durations, events, preds[3])
            try:
                train_risk_scores, train_durations, train_events = self._get_train_risk_scores(
                    model_name, model_path, scenario_enum)
                baselines[model_name] = fit_breslow_baseline_hazard(
                    train_risk_scores, train_durations, train_events)
            except Exception as e:
                self.log(f"Error fitting baseline hazard for {model_name}: {e}")
                baselines[model_name] = None
        return predictions, baselines

    def _group_metrics(self, y_train, train_max, positions, durations, events, risk_scores,
                       baseline, native_prob_fn, n_bootstrap):
        """C-index / integrated Brier / mean AUC for one model inside one
        group, each with a within-group bootstrap percentile interval.

        The group's patients are resampled with replacement; the training
        censoring reference stays fixed, as it does for the cohort-level
        intervals (see pkgs/data_analysis/bootstrap_ci.py)."""
        group_durations = np.asarray(durations, dtype=np.float64)[positions]
        group_events = np.asarray(events).astype(bool)[positions]
        group_risk = np.asarray(risk_scores, dtype=np.float64)[positions]

        point = discrimination_metrics(y_train, train_max, group_risk, group_durations,
                                       group_events, baseline, self._subset_prob_fn(
                                           native_prob_fn, positions),
                                       auc_horizon_days=EVAL_HORIZON_DAYS)

        times = np.linspace(1, EVAL_HORIZON_DAYS, 50)
        survival_probs, brier_source = survival_prob_matrix(
            group_risk, times, baseline, self._subset_prob_fn(native_prob_fn, positions))
        d, e, _ = _administratively_censor(group_durations, group_events, train_max)
        auc_max = min(float(np.max(d)), EVAL_HORIZON_DAYS - 1)
        auc_times = bootstrap_ci.bootstrap_auc_grid(auc_max)
        indices = bootstrap_ci.bootstrap_indices(len(positions), n_bootstrap)
        replicates = bootstrap_ci.bootstrap_model_metrics(
            y_train, d, e, group_risk, survival_probs, times, auc_times, indices)

        return {
            'brier_source': brier_source,
            'c_index': bootstrap_ci.summarize(point['c_index'], replicates['c_index']),
            'brier': bootstrap_ci.summarize(point['brier'], replicates['brier']),
            'auc': bootstrap_ci.summarize(point['auc'], replicates['auc']),
        }

    @staticmethod
    def _subset_prob_fn(native_prob_fn, positions):
        """Restrict a model's native P(event by t) to this group's patients.
        The model's own native_prob_fn returns a value per COHORT patient in
        the canonical order, so a group's curve is that array indexed by the
        group's positions — not a re-prediction."""
        if native_prob_fn is None:
            return None

        def subset(horizon_days):
            values = native_prob_fn(horizon_days)
            if values is None:
                return None
            return np.asarray(values, dtype=np.float64)[positions]

        return subset

    def save_scenario_report(self, scenario_name):
        lines = self.scenario_report_lines.get(scenario_name, [])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = [
            "SUBGROUP PERFORMANCE ANALYSIS REPORT",
            "=" * 80,
            f"Generated on: {timestamp}",
            f"Repetition: {current_rep}",
            f"Scenario: {scenario_name}",
            "",
            "This reports model performance WITHIN demographic groups. It is not a fairness",
            "evaluation: no fairness criterion is defined or tested here, and a between-group",
            "difference in these numbers has causes (group size, event rate, follow-up length,",
            "measurement density, coding practice) that these numbers cannot separate.",
            "",
            "Age groups are bins of anchor_age — the patient's age in their MIMIC anchor year,",
            "NOT their age at the prediction landmark, which the exported scenario data does not",
            "carry enough information to derive. Race is collapsed from admissions.csv ethnicity",
            "and keeps an explicit unknown/not-recorded group. See",
            "pkgs/data_analysis/patient_metadata.py for both rules and why they differ from",
            "demographics.py's.",
            "=" * 80,
            "",
        ]
        report_path = self.output_dir / f'{scenario_name}_subgroup_performance_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(header + lines))
        print(f"Subgroup report saved to: {report_path}")
        self.current_scenario = None

    def create_metrics_comparison_charts(self):
        """No cross-scenario bar charts for subgroup analysis: a grouped bar
        chart over model x scenario x group x metric is unreadable, and the
        per-group numbers with intervals belong in the report where the counts
        that qualify them are visible. Overridden to a no-op so the shared
        runner can call it unconditionally."""
        return
