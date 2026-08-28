"""
Stage 3.0 feature-importance analysis, scoped to four_features/eight_features
only, for whatever CKD_REP is currently set (rep1 for Stage 3.0, rep2-4 for
Stage 3.1) — not rep99.

Mirrors pkgs/scripts/run_stage21_feature_importance.py's rationale (a scoped
driver instead of the module's own entry point), but that script hardcodes
`assert current_rep == 99` and unconditionally also calls
analyze_twenty_features(), which would either crash (rep1 has no rep99 data)
or, once it stopped crashing, violate Stage 3.0's scenario-ordering rule by
touching twenty_features_heterogeneous before it's approved. This variant
drops both restrictions: any rep, four_features/eight_features only.

Usage: PYTHONPATH=. CKD_REP=1 python -m pkgs.scripts.run_stage3_0_four_eight_feature_importance
"""
from pkgs.commons import generate_data_path_latest_rep, current_rep
from pkgs.data_analysis.feature_importance_analysis import FeatureImportanceAnalyzer


def main():
    output_dir = generate_data_path_latest_rep
    analyzer = FeatureImportanceAnalyzer("four_eight_features", output_dir)

    analyzer.log("Starting Stage 3.0 Feature Importance Analysis (four_features/eight_features only)...")
    analyzer.log(f"Output directory: {output_dir}")
    analyzer.log(f"Current repetition: {current_rep}")
    analyzer.log(f"Models to analyze: {', '.join(analyzer.models)}")
    analyzer.log("")

    analyzer.analyze_four_features()
    analyzer.analyze_eight_features()

    analyzer.log("="*80)
    analyzer.log("ANALYSIS COMPLETE")
    analyzer.log("="*80)


if __name__ == "__main__":
    main()
