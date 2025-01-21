import torch
import numpy as np
from torch.utils.data import DataLoader

from exp_common import generate_sample_data, LongitudinalDataset
from exp_common import batch_size, input_dim, hidden_dims, num_risks, time_bins, learning_rate, num_epochs, calculate_c_index, survival_loss
from model_transformer import DynamicTransformer

# Generate data
df = generate_sample_data(num_subjects=200, max_observations=30)

print(f'data \n{df.head()}')
dataset = LongitudinalDataset(df)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
model = DynamicTransformer(
    input_dim=input_dim,
    hidden_dim=hidden_dims[0],
    num_risks=num_risks,
    time_bins=time_bins,
    num_heads=3,
    num_transformer_layers=2,
    dropout_rate=0.2
)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, mask, time_intervals, event_indicators in dataloader:
        optimizer.zero_grad()
        hazard_preds = model(features, mask)
        loss = survival_loss(hazard_preds, time_intervals, event_indicators, num_risks)
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
        hazard_preds = model(features, mask)
        c_indices = calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks)
        all_c_indices.append(c_indices)

# Aggregate results
avg_c_indices = np.mean(all_c_indices, axis=0)
for risk_idx, c_index in enumerate(avg_c_indices):
    print(f"Risk {risk_idx + 1} C-index: {c_index:.4f}")
