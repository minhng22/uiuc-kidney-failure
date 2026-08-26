"""
TEMPORARY / interim rep1 clinical-validity analysis (calibration + DCA), user
request 2026-08-26.

Same rationale as run_rep1_temp_feature_importance.py: rep1's full Stage 3.0
run isn't finished (dynamic_deephit/hazard_transformer/rnn_surv are still
training twenty_features_heterogeneous). ClinicalValidityAnalyzer.analyze_scenario()
already skips (doesn't crash on) any model whose file doesn't exist yet, so
this naturally produces results only for what's actually done:
  - cox, logistic_hazard: all 3 scenarios (fully done)
  - ddh, hazard_transformer, rnn_surv: four_features + eight_features only

Not the final Stage 2.1-style analysis -- re-run once rep1 fully completes.

Scoped driver per CLAUDE.md "Check a script's actual entry point before
running it as an experiment" -- mirrors run_stage21_clinical_validity.py but
asserts CKD_REP=1 instead of 99.

Usage: CKD_REP=1 PYTHONPATH=. python -m pkgs.scripts.run_rep1_temp_clinical_validity
"""
from pkgs.commons import generate_data_path_latest_rep, current_rep
from pkgs.data_analysis.clinical_validity_analysis import ClinicalValidityAnalyzer


def main():
    assert current_rep == 1, f"This driver is scoped to rep1; got CKD_REP={current_rep}."

    output_dir = generate_data_path_latest_rep
    analyzer = ClinicalValidityAnalyzer(output_dir)

    print("Starting TEMPORARY/interim rep1 clinical-validity analysis (calibration + DCA)...")
    print("(dynamic_deephit/hazard_transformer/rnn_surv's twenty_features_heterogeneous")
    print(" training is still in progress -- those combos will be skipped below.)")
    print(f"Output directory: {output_dir}")
    print(f"Current repetition: {current_rep}")

    analyzer.analyze_four_features()
    analyzer.analyze_eight_features()
    analyzer.analyze_twenty_features()

    analyzer.create_metrics_comparison_charts()

    print("=" * 80)
    print("TEMPORARY ANALYSIS COMPLETE (re-run once rep1 fully finishes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
