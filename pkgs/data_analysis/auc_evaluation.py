"""Experiment AUC reporting through the clinical-validity metric implementation."""
import numpy as np

from pkgs.data_analysis.patient_outcomes import (
    patient_level_outcomes, align_predictions_to_patients,
)


def _terminal_frame(frame):
    if 'subject_id' in frame:
        return patient_level_outcomes(frame)
    # Some flat-model callers have already selected features/outcomes and
    # dropped subject_id. Those frames contain one row per patient.
    return frame[['duration_in_days', 'has_esrd']].reset_index(drop=True)


def _report_auc(train_terminal, test_terminal, risk_scores):
    # Runtime import avoids a cycle: clinical validity imports the KFRE
    # experiment for its cache path, and experiments import this adapter.
    from pkgs.data_analysis.clinical_validity_analysis import (
        build_censoring_reference, evaluation_time_cap, mean_time_dependent_auc,
    )

    try:
        y_train = build_censoring_reference(train_terminal)
        cap, _ = evaluation_time_cap(train_terminal)
        value = mean_time_dependent_auc(
            y_train, cap, risk_scores,
            test_terminal['duration_in_days'].to_numpy(dtype=float),
            test_terminal['has_esrd'].to_numpy(dtype=bool),
        )
    except ValueError as exc:
        print(f'Mean time-dependent AUC unavailable: {exc}')
        return None
    print(f'Mean time-dependent AUC (patient-level, supported follow-up below 730 days): {value:.4f}')
    return value


def report_auc(train_frame, test_frame, risk_scores):
    """Score row-aligned risk values, retaining each patient's terminal row.

    Flat model frames are already patient-level. Raw time-varying frames are
    reduced along with their scores so lab rows never become separate patients.
    """
    scores = np.asarray(risk_scores).reshape(-1)
    if len(scores) != len(test_frame):
        raise ValueError('AUC risk scores must align with the supplied test rows')
    if 'subject_id' in test_frame:
        ordered = test_frame[['subject_id', 'duration_in_days']].copy()
        ordered['_position'] = np.arange(len(ordered))
        terminal_rows = (ordered.sort_values(['subject_id', 'duration_in_days'], kind='mergesort')
                        .groupby('subject_id', sort=True).tail(1))
        scores = scores[terminal_rows['_position'].to_numpy()]
    return _report_auc(_terminal_frame(train_frame), _terminal_frame(test_frame), scores)


def report_prediction_auc(model, scenario, train_frame, test_frame):
    """Use the neural model's same patient predictions as clinical validity."""
    import torch

    # predictions() uses CPU inputs, just as when analysis loads a checkpoint.
    # Training may have left the live model on a GPU.
    if isinstance(model, torch.nn.Module):
        device = next(model.parameters()).device
        was_training = model.training
        try:
            model.cpu()
            risk_scores, durations, events, _ = model.predictions(scenario)
        finally:
            model.to(device)
            model.train(was_training)
    else:
        risk_scores, durations, events, _ = model.predictions(scenario)
    terminal = patient_level_outcomes(test_frame)
    aligned, reason = align_predictions_to_patients(terminal, durations, events, type(model).__name__)
    if not aligned:
        raise ValueError(reason)
    return _report_auc(_terminal_frame(train_frame), terminal, risk_scores)
