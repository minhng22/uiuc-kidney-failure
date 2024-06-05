import pandas as pd
from commons import (
    diagnose_icd_file_path, patients_file_path, get_esrd_codes,
    figs_path, figs_path_gender_statistics, figs_path_age_statistics,
    age_bins, figs_path_race_statistics, figs_path_race_stats, figs_path_icd_stats, get_target_esrd_codes,
    get_target_ckd_codes, admissions_file_path
)
import matplotlib.pyplot as plt
import os
import cmocean

def analyze_esrd():
    patients_df, diagnoses_df = get_esrd_patients_and_diagnoses()
    plot_icd_codes(diagnoses_df)

    age_statistics(patients_df, diagnoses_df)
    gender_statistics(patients_df, diagnoses_df)
    race_statistics(patients_df, diagnoses_df)


def age_statistics(patients_df, diagnoses_df):
    print(
        f"age statistics:\n"
        f"mean: {patients_df['anchor_age'].mean():.3f}, std: {patients_df['anchor_age'].std():.3f}, min: {patients_df['anchor_age'].min()}, max: {patients_df['anchor_age'].max()}"
    )

    merged_df = pd.merge(patients_df, diagnoses_df, on='subject_id')

    labels = ['0-16', '17-30', '31-45', '45+']

    merged_df['age_group'] = pd.cut(merged_df['anchor_age'], bins=age_bins, labels=labels, right=False)

    # Group by icd_code and age_group and count occurrences
    grouped = merged_df.groupby(['icd_code', 'age_group'], observed=False).size().reset_index(name='count')

    # Calculate the percentage of each age group within each icd_code
    grouped['percentage'] = grouped.groupby('icd_code')['count'].transform(lambda x: x / x.sum() * 100)

    # Pivot the dataframe to get age groups as columns
    pivot_df = grouped.pivot(index='icd_code', columns='age_group', values='percentage').fillna(0)

    ax = pivot_df.plot(kind='bar', stacked=True)
    ax.set_ylabel('Percentage')
    ax.set_title('Proportion Of Age Groups By ICD Code')
    plt.legend(title='Age Group')

    if not os.path.exists(figs_path):
        os.mkdir(figs_path)
    plt.savefig(figs_path_age_statistics, bbox_inches="tight")
    plt.clf()


def plot_icd_codes(diagnoses_df):
    icd_code_counts = diagnoses_df['icd_code'].value_counts()

    plt.figure(figsize=(10, 7))
    icd_code_counts.plot.bar()

    # Add numbers on top of each bar
    for i, count in enumerate(icd_code_counts):
        plt.text(
            i,
            count,
            f'{count}',
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.title('Proportion Of ICD Codes (By Number of Diagnoses)')
    plt.ylabel('')  # Hide the y-label for the pie chart

    plt.savefig(figs_path_icd_stats, bbox_inches="tight")
    plt.clf()


def gender_statistics(patients_df, diagnoses_df):
    print(
        f"gender statistics:\n"
        f"{patients_df['gender'].value_counts()}."
    )

    merged_df = pd.merge(patients_df, diagnoses_df, on='subject_id')

    grouped = merged_df.groupby(['icd_code', 'gender']).size().reset_index(name='count')
    grouped['percentage'] = grouped.groupby('icd_code')['count'].transform(lambda x: x / x.sum() * 100)
    pivot_df = grouped.pivot(index='icd_code', columns='gender', values='percentage').fillna(0)

    ax = pivot_df.plot(kind='bar', stacked=True)
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage Of Genders For Each ICD Code')
    plt.legend(title='Gender')

    if not os.path.exists(figs_path):
        os.mkdir(figs_path)
    plt.savefig(figs_path_gender_statistics, bbox_inches="tight")
    plt.clf()


# @with_processed_race - if True: (1) filters out patients with selection 'PATIENT DECLINED TO ANSWER', 'UNABLE TO OBTAIN', 'UNKNOWN'
# (2) merge the sub options: 'ASIAN - ASIAN INDIAN' -> 'ASIAN'
def get_admission_df(with_processed_race: bool):
    admission_df = pd.read_csv(admissions_file_path)

    if with_processed_race:
        bad_record_admission_df = admission_df[
            admission_df['race'].isin(["PATIENT DECLINED TO ANSWER", "UNABLE TO OBTAIN", "UNKNOWN"])]
        percentage_filtered = (len(bad_record_admission_df) / len(admission_df)) * 100

        print(f"percentage of patients with race selection 'PATIENT DECLINED TO ANSWER', "
              f"'UNABLE TO OBTAIN', or 'UNKNOWN': {percentage_filtered:.2f}%")
        admission_df = admission_df[
            ~admission_df['race'].isin(["PATIENT DECLINED TO ANSWER", "UNABLE TO OBTAIN", "UNKNOWN"])]
        # merge the sub options: 'ASIAN - ASIAN INDIAN' -> 'ASIAN'
        admission_df['race'] = admission_df['race'].str.split(' - ').str[0]

    return admission_df


def race_statistics(patients_df, diagnoses_df):
    admission_df = get_admission_df(True)

    merged_df = pd.merge(patients_df, admission_df, on='subject_id')
    merged_df = pd.merge(merged_df, diagnoses_df, on='subject_id')
    grouped = merged_df.groupby(['icd_code', 'race']).size().reset_index(name='count')
    grouped['percentage'] = grouped.groupby('icd_code')['count'].transform(lambda x: x / x.sum() * 100)
    pivot_df = grouped.pivot(index='icd_code', columns='race', values='percentage').fillna(0)

    pivot_df.to_csv(figs_path_race_stats)

    # 40 distinct colors
    cmocean_cmap = cmocean.cm.phase
    color_blind_palette = [cmocean_cmap(i / len(pivot_df.columns)) for i in range(len(pivot_df.columns))]

    ax = pivot_df.plot(kind='bar', stacked=True, color=color_blind_palette, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Percentage')
    ax.set_title('Proportion of Races For Each ICD Code')
    plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')

    if not os.path.exists(figs_path):
        os.mkdir(figs_path)

    plt.savefig(figs_path_race_statistics, bbox_inches="tight")
    plt.clf()


def get_esrd_patients_and_diagnoses():
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)

    esrd_diagnose_df = filter_ckd_esrd_diagnose(diagnoses_df)
    esrd_diagnose_df = esrd_diagnose_df[esrd_diagnose_df['icd_code'].isin(get_esrd_codes())]
    print(
        f"number of ESRD subjects: {esrd_diagnose_df['subject_id'].nunique()}\n"
        f"percentage of subjects in dataset: {esrd_diagnose_df['subject_id'].nunique() / diagnoses_df['subject_id'].nunique() * 100:.3f}"
    )

    patients_df = pd.read_csv(patients_file_path)
    patients_df = patients_df[patients_df['subject_id'].isin(esrd_diagnose_df['subject_id'].unique())]

    print(f"number of subjects (for validation): {patients_df['subject_id'].nunique()}")

    return patients_df, esrd_diagnose_df


def filter_ckd_esrd_diagnose(df):
    subject_ids = df.groupby('subject_id').filter(lambda x: any(x['icd_code'].isin(get_target_esrd_codes())) and any(x['icd_code'].isin(get_target_ckd_codes())))[
        'subject_id'].unique()
    print(f"number of patients who have esrd (progressed from ckd) are {len(subject_ids)}")
    return df[df['subject_id'].isin(subject_ids)]


if __name__ == '__main__':
    analyze_esrd()