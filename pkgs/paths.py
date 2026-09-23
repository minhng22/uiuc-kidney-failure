"""Repetition directory names without importing repetition-bound data paths."""


def repetition_directory_name(rep):
    """Production reps use rep_1–rep_5; existing rep99/rep100 keep their names."""
    rep = int(rep)
    return f'rep_{rep}' if 1 <= rep <= 5 else f'rep{rep}'
