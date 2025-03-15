from pkgs.commons import egfr_tv_hazard_transformer_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr, mini
from pkgs.models.hazard_transformer import HazardTransformer
import torch
from torch.utils.data import DataLoader
from pkgs.playground.exp_common import RNNAttentionDataset, batch_size, input_dim, hidden_dims, time_bins, learning_rate, calculate_c_index, survival_loss
import numpy as np
import os

def run_transformer():
    df, df_test = get_train_test_data_egfr(True)
    df = mini(df)
    dataset = RNNAttentionDataset(df, multiple_risk=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_risks = 1
    num_epochs = 10
    model = HazardTransformer(input_dim, hidden_dims[0], time_bins, num_risks)
    if os.path.exists(egfr_tv_hazard_transformer_model_path):
        print("Loading from saved weights")
        model.load_state_dict(torch.load(egfr_tv_hazard_transformer_model_path, map_location=torch.device('cpu'), weights_only=True))
    else:
        print("Start training")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for features, mask, time_intervals, event_indicators in dataloader:
                optimizer.zero_grad()
                hazard_preds, _ = model(features, mask)
                loss = survival_loss(hazard_preds, time_intervals, event_indicators, num_risks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
        torch.save(model.state_dict(), egfr_tv_hazard_transformer_model_path)
        print("Training complete.")
    model.eval()
    all_c_indices = []
    with torch.no_grad():
        for features, mask, time_intervals, event_indicators in dataloader:
            hazard_preds, _ = model(features, mask)
            c_indices = calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks)
            all_c_indices.append(c_indices)
    avg_c_indices = np.mean(all_c_indices, axis=0)
    for risk_idx, c_index in enumerate(avg_c_indices):
        print(f"Risk {risk_idx+1} C-index: {c_index:.2f}")
    test_dataset = RNNAttentionDataset(df_test, multiple_risk=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_c_indices = []
    with torch.no_grad():
        for features, mask, time_intervals, event_indicators in test_dataloader:
            hazard_preds, _ = model(features, mask)
            c_indices = calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks)
            test_c_indices.append(c_indices)
    avg_test_c_indices = np.mean(test_c_indices, axis=0)
    for risk_idx, c_index in enumerate(avg_test_c_indices):
        print(f"Risk {risk_idx+1} Test C-index: {c_index:.2f}")

if __name__ == '__main__':
    run_transformer()
