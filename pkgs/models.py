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
median_survival_times = cph.predict_median(df)
print(median_survival_times)

# Plot the survival functions
cph.plot()
