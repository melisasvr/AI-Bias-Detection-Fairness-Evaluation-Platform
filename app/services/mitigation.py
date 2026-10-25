"""Exploratory global-threshold trade-offs on user-designated validation data."""
import numpy as np
import pandas as pd
from app.schemas import EvaluationConfig
from app.services.metrics import evaluate
from app.utils.validation import validate_data


def threshold_sweep(data: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    """Return a comparison, never select or deploy a threshold automatically."""
    if not config.score:
        raise ValueError('Threshold exploration requires a probability column.')
    frame = validate_data(data, config)
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        candidate = frame.copy()
        candidate[config.prediction] = (candidate[config.score] >= threshold).astype(int)
        report = evaluate(candidate, config)
        gaps = [v['disparities']['equalized_odds_difference'] for v in report['assessments']]
        rows.append(dict(threshold=float(threshold), accuracy=float((candidate[config.target] == candidate[config.prediction]).mean()),
                         worst_equalized_odds_gap=max(gaps) if all(g is not None for g in gaps) else None))
    return pd.DataFrame(rows)
