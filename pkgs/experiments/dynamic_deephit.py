from pkgs.commons import egfr_tv_dynamic_deep_hit_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr, mini
from pkgs.models.dynamicdeephit import DynamicDeepHit
import torch
from torch.utils.data import DataLoader

from pkgs.playground.exp_common import RNNAttentionDataset
from pkgs.playground.exp_common import batch_size, input_dim, hidden_dims, time_bins, learning_rate, calculate_c_index, survival_loss
from pkgs.models.dynamicdeephit import DynamicDeepHit
from pkgs.experiments.utils import evaluate_rnn_model
import numpy as np
import os


def run_ddh():
    df, df_test = get_train_test_data_egfr(True)
    df = mini(df)

    dataset = RNNAttentionDataset(df, multiple_risk=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_risks = 1

    model = DynamicDeepHit(input_dim, hidden_dims, num_risks, time_bins)

    if os.path.exists(egfr_tv_dynamic_deep_hit_model_path):
        print("Loading existing model for testing...")
        model.load_state_dict(torch.load(egfr_tv_dynamic_deep_hit_model_path, weights_only=True))
        model.eval()
        print("Model loaded successfully.")

        evaluate_rnn_model(model, df_test, ['duration_in_days', 'egfr'])

    else:
        print("No saved model found. Starting training...")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        num_epochs = 2

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for features, mask, time_intervals, event_indicators in dataloader:
                optimizer.zero_grad()
                hazard_preds, _ = model(features, mask)
                loss = survival_loss(hazard_preds, time_intervals, event_indicators, num_risks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), egfr_tv_dynamic_deep_hit_model_path)
        print("Training complete.")

        model.eval()
        evaluate_rnn_model(model, df_test, ['duration_in_days', 'egfr'])


if __name__ == '__main__':
    run_ddh()