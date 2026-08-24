"""
Stage 2.1 additional analyses (calibration + decision-curve analysis) scoped driver.

Mirrors pkgs/scripts/run_stage21_feature_importance.py's rationale: a scoped
driver instead of any module's __main__/run_all(), per CLAUDE.md "Check a
script's actual entry point before running it as an experiment" — avoids
touching egfr_components/fivelabms/CKD_FIFTY_FEATURES_HETEROGENEOUS, which are
off-scope here and (for the fifty-features scenario) have no rep99 data.

Must run with CKD_REP=99 (the only rep with trained models for the 3 new
scenarios so far).

Usage: CKD_REP=99 PYTHONPATH=. python -m pkgs.scripts.run_stage21_clinical_validity
"""
from pkgs.commons import generate_data_path_latest_rep, current_rep
from pkgs.data_analysis.clinical_validity_analysis import ClinicalValidityAnalyzer


def main():
    assert current_rep == 99, (
        f"Stage 2.1 must run against rep99 (the only rep with trained models "
        f"for four_features/eight_features/twenty_features_heterogeneous so "
        f"far); got CKD_REP={current_rep}. Set CKD_REP=99."
    )

    output_dir = generate_data_path_latest_rep
    analyzer = ClinicalValidityAnalyzer(output_dir)

    print("Starting Stage 2.1 clinical-validity analysis (calibration + DCA, new scenarios only)...")
    print(f"Output directory: {output_dir}")
    print(f"Current repetition: {current_rep}")

    analyzer.analyze_four_features()
    analyzer.analyze_eight_features()
    analyzer.analyze_twenty_features()

    analyzer.create_metrics_comparison_charts()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
