import pandas as pd


def filter_df_on_icd_code(df: pd.DataFrame, arr_1: list, arr_2: list) -> pd.DataFrame:
    """
    Filter dataframe to keep only patients who have at least one ICD code from arr_1 
    AND at least one ICD code from arr_2.
    
    Args:
        df: DataFrame with 'subject_id' and 'icd_code' columns
        arr_1: List of ICD codes (first condition)
        arr_2: List of ICD codes (second condition)
    
    Returns:
        Filtered DataFrame with only patients having codes from both arrays
    """
    # Find patients with at least one code from arr_1
    patients_with_arr1 = df[df['icd_code'].isin(arr_1)]['subject_id'].unique()
    
    # Find patients with at least one code from arr_2
    patients_with_arr2 = df[df['icd_code'].isin(arr_2)]['subject_id'].unique()
    
    # Patients who have codes from both arrays
    patients_with_both = set(patients_with_arr1) & set(patients_with_arr2)
    
    # Filter to keep only those patients
    result = df[df['subject_id'].isin(patients_with_both)]
    
    return result
