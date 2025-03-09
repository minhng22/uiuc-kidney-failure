import torch
import numpy as np
from torch.utils.data import DataLoader

from exp_common import generate_sample_data, LongitudinalDataset
from exp_common import batch_size, input_dim, hidden_dims, num_risks_multiple_risks, time_bins, learning_rate, num_epochs, calculate_c_index, survival_loss
from pkgs.models.dynamicdeephit import DynamicDeepHit

# Generate data
df = generate_sample_data(num_subjects=10000, max_observations=30)

print(f'data \n{df.head()}')
dataset = LongitudinalDataset(df)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
model = DynamicDeepHit(input_dim, hidden_dims, num_risks_multiple_risks, time_bins)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, mask, time_intervals, event_indicators in dataloader:
        optimizer.zero_grad()
        hazard_preds, _ = model(features, mask)
        loss = survival_loss(hazard_preds, time_intervals, event_indicators, num_risks_multiple_risks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

print("Training complete.")

# Evaluation loop with C-index
model.eval()
all_c_indices = []

with torch.no_grad():
    for features, mask, time_intervals, event_indicators in dataloader:
        hazard_preds, _ = model(features, mask)
        c_indices = calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks_multiple_risks)
        all_c_indices.append(c_indices)

# Aggregate results
avg_c_indices = np.mean(all_c_indices, axis=0)
for risk_idx, c_index in enumerate(avg_c_indices):
    print(f"Risk {risk_idx + 1} C-index: {c_index:.4f}")

# Evaluation loop with C-index on test data
model.eval()
test_c_indices = []

# Generate testing data
test_df = generate_sample_data(num_subjects=10000, max_observations=20, seed=123)  # Different seed for testing
print(f"Testing data:\n{test_df.head()}")
test_dataset = LongitudinalDataset(test_df)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for features, mask, time_intervals, event_indicators in test_dataloader:
        hazard_preds, _ = model(features, mask)
        c_indices = calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks_multiple_risks)
        test_c_indices.append(c_indices)

# Aggregate results
avg_test_c_indices = np.mean(test_c_indices, axis=0)
for risk_idx, c_index in enumerate(avg_test_c_indices):
    print(f"Risk {risk_idx + 1} Test C-index: {c_index:.4f}")