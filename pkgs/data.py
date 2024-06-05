import pandas as pd
from commons import (
    diagnose_icd_file_path, patients_file_path, get_kidney_failure_codes,
    figs_path, figs_path_gender_statistics, figs_path_age_statistics,
    age_bins, figs_path_race_statistics, figs_path_race_stats, figs_path_icd_stats
)
import matplotlib.pyplot as plt
import os
import cmocean

def analysis_diagnose_icd():
    patients_df, diagnoses_df = get_kidney_failure_patients_and_diagnoses()
    print("analyzing patients")

    #gender_statistics(patients_df, diagnoses_df)
    #age_statistics(patients_df, diagnoses_df)
    #race_statistics(patients_df, diagnoses_df)
    plot_icd_code_pie_chart(diagnoses_df)


def age_statistics(patients_df, diagnoses_df):
    print(
        f"Age statistics:\n"
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
    ax.set_title('Percentage of Age Groups by ICD Code')
    plt.legend(title='Age Group')

    if not os.path.exists(figs_path):
        os.mkdir(figs_path)
    plt.savefig(figs_path_age_statistics, bbox_inches="tight")
    plt.clf()


def plot_icd_code_pie_chart(diagnoses_df):
    # Count the occurrences of each ICD code
    icd_code_counts = diagnoses_df['icd_code'].value_counts()

    # Create a pie chart
    plt.figure(figsize=(10, 7))
    icd_code_counts.plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False
    )

    plt.title('Distribution of ICD Codes')
    plt.ylabel('')  # Hide the y-label for the pie chart

    plt.savefig(figs_path_icd_stats, bbox_inches="tight")
    plt.clf()


def gender_statistics(patients_df, diagnoses_df):
    print(
        f"Gender statistics:\n"
        f"{patients_df['gender'].value_counts()}."
    )

    merged_df = pd.merge(patients_df, diagnoses_df, on='subject_id')

    grouped = merged_df.groupby(['icd_code', 'gender']).size().reset_index(name='count')
    grouped['percentage'] = grouped.groupby('icd_code')['count'].transform(lambda x: x / x.sum() * 100)
    pivot_df = grouped.pivot(index='icd_code', columns='gender', values='percentage').fillna(0)

    ax = pivot_df.plot(kind='bar', stacked=True)
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage of Genders for each ICD Code')
    plt.legend(title='Gender')

    if not os.path.exists(figs_path):
        os.mkdir(figs_path)
    plt.savefig(figs_path_gender_statistics, bbox_inches="tight")
    plt.clf()


# @with_processed_race - if True: (1) filters out patients with selection 'PATIENT DECLINED TO ANSWER', 'UNABLE TO OBTAIN', 'UNKNOWN'
# (2) merge the sub options: 'ASIAN - ASIAN INDIAN' -> 'ASIAN'
def get_admission_df(with_processed_race: bool):
    admission_df = pd.read_csv('../data/mimic-iv-2.2/hosp/admissions.csv')

    if with_processed_race:
        bad_record_admission_df = admission_df[
            admission_df['race'].isin(["PATIENT DECLINED TO ANSWER", "UNABLE TO OBTAIN", "UNKNOWN"])]
        percentage_filtered = (len(bad_record_admission_df) / len(admission_df)) * 100

        print(f"Percentage of patients with race selection 'PATIENT DECLINED TO ANSWER', "
              f"'UNABLE TO OBTAIN', or 'UNKNOWN': {percentage_filtered:.2f}%")
        print(f"Filtering out those records")
        admission_df = admission_df[
            ~admission_df['race'].isin(["PATIENT DECLINED TO ANSWER", "UNABLE TO OBTAIN", "UNKNOWN"])]

        print(f"merging")
        # merge the sub options: 'ASIAN - ASIAN INDIAN' -> 'ASIAN'
        admission_df['race'] = admission_df['race'].str.split(' - ').str[0]

    return admission_df


def race_statistics(patients_df, diagnoses_df):
    print(f"Race statistics:\n")

    admission_df = get_admission_df(False)

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
    ax.set_title('Percentage of Races for each ICD Code')
    plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')

    if not os.path.exists(figs_path):
        os.mkdir(figs_path)

    plt.savefig(figs_path_race_statistics, bbox_inches="tight")
    plt.clf()


def get_kidney_failure_patients_and_diagnoses():
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)
    print(
        f"number of rows: {diagnoses_df.shape[0]}. number of subjects: {diagnoses_df['subject_id'].nunique()}"
    )

    print("filtering for kidney failure")
    k = get_kidney_failure_codes()
    kidney_failure_diagnose_df = diagnoses_df[diagnoses_df['icd_code'].isin(k)]
    print(
        f"number of rows: {kidney_failure_diagnose_df.shape[0]}. number of subjects: {kidney_failure_diagnose_df['subject_id'].nunique()}\n"
        f"percentage of subjects in dataset with kidney failure: {kidney_failure_diagnose_df['subject_id'].nunique() / diagnoses_df['subject_id'].nunique() * 100:.3f}"
    )

    kf_subjects = kidney_failure_diagnose_df['subject_id'].unique()
    print(f"number of subjects with kidney failure (for validation): {len(kf_subjects)}")

    patients_df = pd.read_csv(patients_file_path)
    patients_df = patients_df[patients_df['subject_id'].isin(kf_subjects)]

    print(
        f"printing number of subjects out for validation\n"
        f"number of rows: {patients_df.shape[0]}. number of subjects: {patients_df['subject_id'].nunique()}"
    )

    return patients_df, kidney_failure_diagnose_df


if __name__ == '__main__':
    analysis_diagnose_icd()