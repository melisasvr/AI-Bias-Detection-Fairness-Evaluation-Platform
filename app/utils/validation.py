"""Strict validation: never silently drop uploaded records."""
import numpy as np
import pandas as pd
from app.schemas import EvaluationConfig

MAX_ROWS = 200_000
MAX_GROUPS = 100

def validate_data(data: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    """Return a normalized copy or raise an actionable ValueError."""
    if data.empty or len(data) > MAX_ROWS:
        raise ValueError(f"Provide between 1 and {MAX_ROWS:,} rows.")
    if data.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    required = [config.target, config.prediction, *config.sensitive]
    if config.score:
        required.append(config.score)
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data[required].copy()
    if frame.isna().any().any():
        raise ValueError("Selected columns contain missing values. Resolve them before evaluation.")
    for col in (config.target, config.prediction):
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f"{col}: labels must be binary 0/1. Map your positive outcome to 1.")
        frame[col] = frame[col].astype(int)
    for col in config.sensitive:
        frame[col] = frame[col].astype(str)
        if frame[col].str.strip().eq('').any():
            raise ValueError(f"{col}: empty group labels are not allowed.")
        if frame[col].str.len().max() > 120:
            raise ValueError("Group labels must be at most 120 characters.")
    if len(frame[list(config.sensitive)].drop_duplicates()) > MAX_GROUPS:
        raise ValueError(f"Too many intersection groups; maximum is {MAX_GROUPS}.")
    if config.score:
        score = pd.to_numeric(frame[config.score], errors='coerce')
        if not np.isfinite(score).all() or not score.between(0, 1).all():
            raise ValueError("Scores must be finite probabilities in [0, 1] for outcome 1.")
        frame[config.score] = score.astype(float)
    return frame
