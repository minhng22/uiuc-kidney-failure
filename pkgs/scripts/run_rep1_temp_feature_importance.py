"""
TEMPORARY / interim rep1 feature-importance analysis (user request, 2026-08-26).

Rep1's full Stage 3.0 run isn't finished yet: dynamic_deephit, hazard_transformer,
and rnn_surv are still training `twenty_features_heterogeneous` in two separate
background processes (see EXPERIMENT_STATUS.md / stage3_0_background_process_log.txt).
This is an interim look at everything that HAS finished so far, not the final
Stage 2.1-style analysis (which will re-run once rep1 fully completes, per
EXPERIMENT_PLAN_DETAILS.md Stage 3.0/2.1).

Relies on FeatureImportanceAnalyzer.analyze_all_models()'s existing behavior of
skipping (not crashing on) any model whose file doesn't exist yet — logged as
"Model file not found: ..." in the per-scenario report — so this naturally
analyzes only the models/scenarios that are actually done:
  - cox, logistic_hazard: all 3 scenarios (fully done)
  - ddh, hazard_transformer, rnn_surv: four_features + eight_features only
    (twenty_features_heterogeneous still training, will be skipped here)

Scoped driver per CLAUDE.md "Check a script's actual entry point before
running it as an experiment" — mirrors run_stage21_feature_importance.py but
asserts CKD_REP=1 instead of 99.

Usage: CKD_REP=1 PYTHONPATH=. python -m pkgs.scripts.run_rep1_temp_feature_importance
"""
from pkgs.commons import generate_data_path_latest_rep, current_rep
from pkgs.data_analysis.feature_importance_analysis import FeatureImportanceAnalyzer


def main():
    assert current_rep == 1, f"This driver is scoped to rep1; got CKD_REP={current_rep}."

    output_dir = generate_data_path_latest_rep
    analyzer = FeatureImportanceAnalyzer("four_eight_twenty_features", output_dir)

    analyzer.log("Starting TEMPORARY/interim rep1 Feature Importance Analysis...")
    analyzer.log("(dynamic_deephit/hazard_transformer/rnn_surv's twenty_features_heterogeneous")
    analyzer.log(" training is still in progress -- those combos will be skipped below,")
    analyzer.log(" not analyzed as absent/failed.)")
    analyzer.log(f"Output directory: {output_dir}")
    analyzer.log(f"Current repetition: {current_rep}")
    analyzer.log(f"Models to analyze: {', '.join(analyzer.models)}")
    analyzer.log("")

    analyzer.analyze_four_features()
    analyzer.analyze_eight_features()
    analyzer.analyze_twenty_features()

    analyzer.log("="*80)
    analyzer.log("TEMPORARY ANALYSIS COMPLETE (re-run once rep1 fully finishes)")
    analyzer.log("="*80)


if __name__ == "__main__":
    main()
