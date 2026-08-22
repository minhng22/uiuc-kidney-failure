"""
Stage 2.1 (feature-importance analysis) scoped driver.

Runs FeatureImportanceAnalyzer for only the 3 new scenarios
(four_features, eight_features, twenty_features_heterogeneous), not
`main()`'s egfr_components/fivelabms analyses — deliberately avoiding
the footgun documented in CLAUDE.md "Check a script's actual entry
point before running it as an experiment": running the module's actual
`__main__`/`main()` would also analyze egfr_components/fivelabms against
whatever CKD_REP is set, which is off-scope for this stage and those
scenarios have no rep99 data/models to analyze anyway.

Must run with CKD_REP=99 (the only rep with trained models for the 3 new
scenarios so far — Stage 3's rep1-5 runs haven't happened yet).

Usage: CKD_REP=99 PYTHONPATH=. python -m pkgs.scripts.run_stage21_feature_importance
"""
from pkgs.commons import generate_data_path_latest_rep, current_rep
from pkgs.data_analysis.feature_importance_analysis import FeatureImportanceAnalyzer


def main():
    assert current_rep == 99, (
        f"Stage 2.1 must run against rep99 (the only rep with trained models "
        f"for four_features/eight_features/twenty_features_heterogeneous so "
        f"far); got CKD_REP={current_rep}. Set CKD_REP=99."
    )

    output_dir = generate_data_path_latest_rep
    analyzer = FeatureImportanceAnalyzer("four_eight_twenty_features", output_dir)

    analyzer.log("Starting Stage 2.1 Feature Importance Analysis (new scenarios only)...")
    analyzer.log(f"Output directory: {output_dir}")
    analyzer.log(f"Current repetition: {current_rep}")
    analyzer.log(f"Models to analyze: {', '.join(analyzer.models)}")
    analyzer.log("")

    analyzer.analyze_four_features()
    analyzer.analyze_eight_features()
    analyzer.analyze_twenty_features()

    analyzer.log("="*80)
    analyzer.log("ANALYSIS COMPLETE")
    analyzer.log("="*80)


if __name__ == "__main__":
    main()
