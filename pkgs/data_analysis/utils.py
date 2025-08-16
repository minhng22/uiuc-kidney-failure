# calculate eGFR using the CKD-EPI 2021 formula
# https://www.mdcalc.com/calc/3939/ckd-epi-equations-glomerular-filtration-rate-gfr#evidence
# based on https://pubmed.ncbi.nlm.nih.gov/34554658/
def calculate_eGFR(row):
    gender = row['gender']
    age = row['anchor_age']
    serum_creatinine = row['valuenum']  # unit of serum creatinine in MIMIC-IV data is already mg/dL.
    assert serum_creatinine != 0, f"bad value of serum_creatinine {row['subject_id']}"
    assert gender in ['M', 'F'], \
        f"bad value of gender {row['subject_id']}"  # range of gender value in mimic-iv dataset.

    def get_ab():
        if gender == 'M':
            if serum_creatinine <= 0.9:
                return 0.9, -0.302
            return 0.9, -1.2
        if serum_creatinine <= 0.7:
            return 0.7, -0.241
        return 0.7, -1.2

    gender_factor = 1 if gender == "M" else 1.012
    A, B = get_ab()

    return 142 * ((serum_creatinine / A) ** B) * (0.9938 ** age) * gender_factor