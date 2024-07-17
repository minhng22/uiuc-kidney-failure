import pandas as pd
from lifelines import CoxPHFitter

# Example dataset
data = {
    'duration': [5, 6, 6, 2.5, 4, 3, 10],
    'event': [1, 1, 0, 1, 0, 0, 1],
    'age': [50, 55, 60, 48, 42, 39, 68],
    'treatment': [0, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Initialize the CoxPHFitter
cph = CoxPHFitter()

# Fit the model
cph.fit(df, duration_col='duration', event_col='event')

# Print the summary
cph.print_summary()

# Predict the median survival time for each individual
censored_df = df.loc[~df['event'].astype(bool)]
M = cph.predict_median(censored_df)
print(f"predict_median\n{M}")

# Plot the survival functions
cph.plot()

D = cph.predict_survival_function(df)
print(f"predict_survival_function\n{D}")


# Example new data
new_data = pd.DataFrame({
    'age': [60, 40, 50],
    'treatment': [0, 0, 1],
    'duration': [5, 6, 6]
})

# Predict survival function for the new data
surv_pred = cph.predict_survival_function(new_data)

# Print the predicted survival probabilities
print(f"predict_survival_function\n{surv_pred}")

M = cph.predict_median(new_data)
print(f"predict_median\n{M}")
