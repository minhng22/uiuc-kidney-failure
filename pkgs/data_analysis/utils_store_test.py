import pandas as pd
from pkgs.data.utils_store import filter_df_on_icd_code

def test_filter_diagnoses_for_patients_with_both_icd_codes():
    test_cases = [
        {
            "input_df": pd.DataFrame({
                'subject_id': [1, 1, 2, 2, 3, 3, 4, 4],
                'icd_code': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'D']
            }),
            "arr_1": ['A'],
            "arr_2": ['C'],
            "expected_output": pd.DataFrame({
                'subject_id': [2, 2],
                'icd_code': ['A', 'C']
            })
        },
        {
            "input_df": pd.DataFrame({
                'subject_id': [1, 1, 2, 2, 3, 3, 4, 4],
                'icd_code': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'D']
            }),
            "arr_1": ['A'],
            "arr_2": ['B'],
            "expected_output": pd.DataFrame({
                'subject_id': [1, 1],
                'icd_code': ['A', 'B']
            })
        },
        {
            "input_df": pd.DataFrame({
                'subject_id': [1, 2, 3, 4],
                'icd_code': ['A', 'B', 'C', 'D']
            }),
            "arr_1": ['A'],
            "arr_2": ['D'],
            "expected_output": pd.DataFrame({
                'subject_id': [],
                'icd_code': []
            })
        }
    ]

    for i, test_case in enumerate(test_cases):
        input_df = test_case["input_df"]
        arr_1 = test_case["arr_1"]
        arr_2 = test_case["arr_2"]
        expected_output = test_case["expected_output"]
        expected_output["subject_id"] = expected_output["subject_id"].astype('int64')
        expected_output["icd_code"] = expected_output["icd_code"].astype('str')

        output_df = filter_df_on_icd_code(input_df, arr_1, arr_2)

        try:
            pd.testing.assert_frame_equal(output_df.reset_index(drop=True), expected_output.reset_index(drop=True))
            print(f"Test case {i + 1} passed.")
        except AssertionError as e:
            print(
                f"Test case {i + 1} failed.\n"
                f"Expected output:\n{expected_output}\n"
                f"Output:\n{output_df}\n"
            )
            print(e)


if __name__ == "__main__":
    test_filter_diagnoses_for_patients_with_both_icd_codes()