import torch


def combine_loss(hazard_preds, time_intervals, event_indicators, num_risks, w1=0.5, w2=0.1):
    batch_size = hazard_preds.size(0)
    num_timepoints = hazard_preds.size(2)

    total_loss = 0

    for risk in range(num_risks):
        risk_hazard_preds = hazard_preds[:, risk, :]
        risk_event_indicators = event_indicators[:, risk]

        time_indices = time_intervals[:, 0].clamp(max=num_timepoints - 1).long()

        event_log_prob = torch.log(risk_hazard_preds[torch.arange(batch_size), time_indices]) * risk_event_indicators

        censor_log_prob = torch.zeros(batch_size, device=risk_hazard_preds.device)
        for i in range(batch_size):
            t = time_indices[i].item()
            if t > 0:
                censor_log_prob[i] = torch.sum(torch.log(1 - risk_hazard_preds[i, :t]))

        censor_log_prob = censor_log_prob * (1 - risk_event_indicators)

        log_likelihood_loss = -torch.mean(event_log_prob + censor_log_prob)

        ranking_loss = 0
        count = 0
        for i in range(batch_size):
            for j in range(batch_size):
                if time_intervals[i] < time_intervals[j] and risk_event_indicators[i] == 1:
                    t_i = time_indices[i].item()
                    F_i = torch.sum(risk_hazard_preds[i, :t_i])
                    F_j = torch.sum(risk_hazard_preds[j, :t_i])
                    ranking_loss += torch.exp(-(F_i - F_j) / w2)
                    count += 1

        if count > 0:
            ranking_loss /= count

        total_loss += log_likelihood_loss + w1 * ranking_loss

    return total_loss / num_risks

# Hyperparameters
input_dim = 2 #['duration_in_days', 'egfr']
hidden_dims = [64, 32]
num_risks_multiple_risks = 2
time_bins = 30
batch_size = 16
learning_rate = 1e-3
num_epochs = 1
