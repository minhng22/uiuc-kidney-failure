#!/usr/bin/env python3
"""Regenerate the Springer Nature section files from the lead PLOS manuscript.

    python "paper/to submit 2026/scripts/plos_to_sections.py"

PLOS requires one self-contained .tex with figures uploaded separately; sn-article.tex
\\inputs sections/{introduction,methods,results,discussion}.tex and embeds figures. This
script converts the PLOS body so the two cannot drift: numbered headings, \\nameref ->
Section~\\ref, \\cite -> \\citep, and each figure's `% FIGFILE: <path>` marker becomes an
\\includegraphics. The ML4H sections (ml4h_*.tex) are a condensed, anonymized cut and are
edited by hand.
"""
import re
from pathlib import Path

CONTENT = Path(__file__).resolve().parents[1] / "paper content"
SECTIONS = {"Introduction": "introduction", "Materials and methods": "methods",
            "Results": "results", "Discussion": "discussion"}
SUPPLEMENT = {
    "S1 Checklist": "the TRIPOD+AI checklist in the Supplementary Information",
    "S1 Fig": "Supplementary Fig.~S1", "S2 Fig": "Supplementary Fig.~S2",
    "S3 Fig": "Supplementary Fig.~S3", "S4 Fig": "Supplementary Fig.~S4",
    "S5 Fig": "Supplementary Fig.~S5", "S1 Table": "Supplementary Table~S1",
    "S2 Table": "Supplementary Table~S2",
}


def convert(text):
    text = re.sub(r"\\(sub)*section\*\{", lambda m: m.group(0).replace("*", ""), text)
    text = text.replace("\\section{Materials and methods}", "\\section{Methods}")
    text = re.sub(r"\\nameref\{([^}]+)\}", r"Section~\\ref{\1}", text)
    text = re.sub(r"\\cite\{", r"\\citep{", text)
    text = text.replace("Fig~\\ref", "Fig.~\\ref")
    text = re.sub(r"^(\s*)% FIGFILE: (\S+)\s*$",
                  r"\1\\includegraphics[width=\\textwidth]{\2}", text, flags=re.M)
    for key, value in SUPPLEMENT.items():
        text = text.replace(key, value)
    return text


def main():
    plos = (CONTENT / "plos_digital_health.tex").read_text()
    body = plos[plos.index("\\section*{Introduction}"):plos.index("\\section*{Supporting information}")]
    parts = re.split(r"(?=^\\section\*\{)", body, flags=re.M)
    header = ("% GENERATED from plos_digital_health.tex by scripts/plos_to_sections.py -- edit the PLOS\n"
              "% file and rerun the script rather than editing this file directly.\n\n")
    for part in parts:
        if not part.strip():
            continue
        title = re.match(r"\\section\*\{([^}]+)\}", part).group(1)
        out = CONTENT / "sections" / f"{SECTIONS[title]}.tex"
        out.write_text(header + convert(part).rstrip() + "\n")
        print(f"wrote {out.relative_to(CONTENT)}")




# ---------------------------------------------------------------------------
# ML4H appendix: the long-form derivation, model, and evaluation text from the PLOS
# manuscript, anonymized (no first-person references to prior work) and with labels
# remapped to the ML4H main text / appendix.
ML4H_LABELS = {"sec:training": "app:training", "sec:eval": "app:evaluation",
               "sec:calibration_results": "sec:results", "sec:limitations": "sec:limitations",
               "sec:kfre": "sec:kfre", "sec:outcome": "sec:outcome", "sec:cohort": "sec:cohort",
               "sec:results": "sec:results"}


def ml4h_convert(text):
    text = text.replace("our prior MIMIC-CKD benchmark", "a prior MIMIC-CKD benchmark")
    text = text.replace("our prior benchmark's", "a prior benchmark's")
    text = text.replace("our prior benchmark", "a prior benchmark")
    text = re.sub(r"\\nameref\{([^}]+)\}",
                  lambda m: f"Section~\\ref{{{ML4H_LABELS.get(m.group(1), m.group(1))}}}", text)
    text = re.sub(r"\\cite\{", r"\\citep{", text)
    text = text.replace("Fig~\\ref", "Figure~\\ref")
    text = re.sub(r"^(\s*)% FIGFILE: (\S+)\s*$",
                  r"\1\\includegraphics[width=\\linewidth]{\2}", text, flags=re.M)
    text = text.replace("S1 Table", "the released per-split results files")
    for key, value in SUPPLEMENT.items():
        text = text.replace(key, value.replace("Supplementary", "supplementary"))
    return text


def between(text, start, end):
    return text[text.index(start):text.index(end)]


def build_ml4h_appendix():
    plos = (CONTENT / "plos_digital_health.tex").read_text()
    old = (CONTENT / "sections" / "ml4h_appendix.tex").read_text()
    kfre = between(old, "\\section{KFRE Coefficients", "\\section{Benchmarked Models")
    cohort = between(plos, "KFRE's inputs are drawn from separate", "\\subsubsection*{Twenty features")
    models = between(plos, "\\subsubsection*{Traditional survival analysis models}",
                     "\\subsection*{Model training and optimization}")
    models = models.replace("\\subsubsection*{", "\\subsection{")
    training = between(plos, "\\subsection*{Model training and optimization}", "\\subsection*{Evaluation}")
    training = training.replace("\\subsection*{Model training and optimization} \\label{sec:training}",
                                "\\subsection{Model Training and Optimization}\\label{app:training}")
    evaluation = between(plos, "\\textbf{Repetitions and aggregation.}", "\\section*{Results}")
    paired = between(plos, "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{2pt}",
                     "\\textbf{Only Dynamic-DeepHit consistently outranks KFRE.}")
    subgroup = between(plos, "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lcccccc@{}}",
                       "Table~\\ref{tab:subgroup} reports")
    calib_fig = between(plos, "\\begin{figure}[h!]\n    \\centering\n    % FIGFILE: figs/four_features_calibration_plot.png",
                        "\\textbf{KFRE underestimates risk")
    dca_fig = between(plos, "\\begin{figure}[h!]\n    \\centering\n    % FIGFILE: figs/four_features_decision_curve_plot.png",
                      "\\textbf{Treat-all is a strong comparator")
    def wide(block, env):
        return block.replace(f"\\begin{{{env}}}", f"\\begin{{{env}*}}").replace(f"\\end{{{env}}}", f"\\end{{{env}*}}")
    paired, subgroup = wide(paired, "table"), wide(subgroup, "table")
    calib_fig, dca_fig = wide(calib_fig, "figure"), wide(dca_fig, "figure")
    out = (
        "% GENERATED by scripts/plos_to_sections.py from plos_digital_health.tex (long-form text) and the\n"
        "% previous KFRE-coefficient appendix -- edit the PLOS file and rerun rather than editing here.\n"
        "\\appendix\n\n"
        "\\section{Cohort Construction: Full Derivation} \\label{app:cohort}\n\n" + cohort.strip() + "\n\n"
        + kfre.strip() + "\n\n"
        "\\section{Benchmarked Models: Full Architectural Detail} \\label{app:models}\n\n" + models.strip()
        + "\n\n" + training.strip() + "\n\n"
        "\\section{Evaluation Definitions and Scope} \\label{app:evaluation}\n\n" + evaluation.strip() + "\n\n"
        "\\section{Additional Results} \\label{app:results}\n\n" + paired.strip() + "\n\n" + subgroup.strip()
        + "\n\n" + calib_fig.strip() + "\n\n" + dca_fig.strip() + "\n"
    )
    out = ml4h_convert(out).replace("[h!]", "[htbp]")
    (CONTENT / "sections" / "ml4h_appendix.tex").write_text(out)
    print("wrote sections/ml4h_appendix.tex")


if __name__ == "__main__":
    main()
    build_ml4h_appendix()
