"""
Stage 3.0 clinical-validity analysis (calibration + decision-curve), scoped
to four_features/eight_features only, for whatever CKD_REP is currently set
(rep1 for Stage 3.0, rep2-4 for Stage 3.1) — not rep99.

Mirrors pkgs/scripts/run_stage21_clinical_validity.py's rationale, but that
script hardcodes `assert current_rep == 99` and unconditionally also calls
analyze_twenty_features(). See run_stage3_0_four_eight_feature_importance.py
for why that's dropped here — this variant runs against any rep and skips
twenty_features_heterogeneous until Stage 3.0's scenario-ordering rule
approves it. create_metrics_comparison_charts() only iterates scenarios
actually analyzed in this run, so the two-scenario charts here are correct
on their own, not partial versions of a three-scenario chart.

Usage: PYTHONPATH=. CKD_REP=1 python -m pkgs.scripts.run_stage3_0_four_eight_clinical_validity
"""
from pkgs.commons import generate_data_path_latest_rep, current_rep
from pkgs.data_analysis.clinical_validity_analysis import ClinicalValidityAnalyzer


def main():
    output_dir = generate_data_path_latest_rep
    analyzer = ClinicalValidityAnalyzer(output_dir)

    print("Starting Stage 3.0 clinical-validity analysis (calibration + DCA, four_features/eight_features only)...")
    print(f"Output directory: {output_dir}")
    print(f"Current repetition: {current_rep}")

    analyzer.analyze_four_features()
    analyzer.analyze_eight_features()

    analyzer.create_metrics_comparison_charts()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
