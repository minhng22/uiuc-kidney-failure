## TODO For PLOS submission


**TODO 1 — reporting improvement for the survival-prediction benchmark**
- Relevant — TRIPOD+AI applies because the hazard and survival models are evaluated for individual prognostic prediction, rather than solely for associations or hazard ratios. Its scope includes both regression and machine learning ([official scope](https://www.tripod-statement.org/scope/)).
- Recommendation: verify reporting completeness: STROBE is already cited for cohort flow. Audit the manuscript against applicable STROBE and TRIPOD+AI items, identify actual omissions, and supply a checklist as appropriate. Treat this as manuscript preparation, not an established methodological defect.

**TODO 2 — complete ethics and data-access reporting before submission**
- Relevant as a reporting fix: the lead [PLOS manuscript](paper/to%20submit%202026/paper%20content/plos_digital_health.tex#L411) has only a commented-out ethics statement; the [Springer version](paper/to%20submit%202026/paper%20content/sn-article.tex#L73) has generic wording. [PLOS submission guidelines](https://journals.plos.org/digitalhealth/s/submission-guidelines#loc-human-subjects-research) call for an ethics statement in Methods identifying approval or explaining why it was not needed, and addressing informed consent.
- Fix (text, before submission): describe the secondary use of de-identified data, cite the source dataset's documented ethics approval and consent waiver, and state the verified approval/exemption basis applicable to this analysis. Distinguish the source dataset's oversight from any determination covering this study; do not assume approval by particular institutions or invent an exemption. This is not, by itself, a finding that new approval or experiments are required.
- Data access and reproducibility: identify the MIMIC-IV version actually used and its DOI, and explain access through PhysioNet's credentialing, required training, and data-use agreement. These are dataset-access conditions; no blanket PLOS requirement to recite the exact CITI course title was found. Include author-specific credential/training claims only when verified.

**Separate bibliography cleanup (not an ethics gap)**
- `ishwaran2008random` lacks a journal field. Check `lee2018deephit` against the selected bibliography style before resolving the reported volume/number warning. `hu2022locf_bias` already has an author and is an arXiv `@misc` entry; the earlier missing-author/publisher claim should not be carried forward without a current check.
