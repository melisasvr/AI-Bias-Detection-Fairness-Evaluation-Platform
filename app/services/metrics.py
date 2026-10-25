"""Fairlearn group assessment with explicit denominator handling."""
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import json
from typing import Any
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, selection_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.schemas import EvaluationConfig
from app.utils.validation import validate_data


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float('nan')


def _tpr(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return _ratio(int(((p == 1) & (y == 1)).sum()), int((y == 1).sum()))


def _fpr(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return _ratio(int(((p == 1) & (y == 0)).sum()), int((y == 0).sum()))


def _gap(series: pd.Series) -> float | None:
    if len(series) < 2 or series.isna().any():
        return None
    return float(series.max() - series.min())


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def evaluate(data: pd.DataFrame, config: EvaluationConfig) -> dict[str, Any]:
    """Assess each marginal attribute and observed intersections on binary labels."""
    frame = validate_data(data, config)
    views = [(col,) for col in config.sensitive]
    if len(config.sensitive) > 1:
        views.append(config.sensitive)
    assessments = []
    for columns in views:
        # JSON tuples prevent collisions from delimiters in user-supplied labels.
        keys = frame[list(columns)].apply(lambda row: json.dumps(row.tolist(), ensure_ascii=False), axis=1)
        y, p = frame[config.target], frame[config.prediction]
        metric_frame = MetricFrame(metrics={
            'accuracy': accuracy_score,
            'precision': partial(precision_score, zero_division=np.nan),
            'recall': partial(recall_score, zero_division=np.nan),
            'f1': partial(f1_score, zero_division=np.nan),
            'selection_rate': selection_rate,
            'true_positive_rate': _tpr,
            'false_positive_rate': _fpr,
        }, y_true=y, y_pred=p, sensitive_features=keys)
        table = metric_frame.by_group.copy()
        calibration = []
        warnings = []
        for key in table.index:
            subset = frame.loc[keys.eq(key)]
            truth = subset[config.target]
            table.loc[key, 'count'] = len(subset)
            table.loc[key, 'positives'] = int(truth.sum())
            table.loc[key, 'negatives'] = int((truth == 0).sum())
            if len(subset) < config.min_group_size:
                warnings.append(f"{key}: only {len(subset)} records; estimates may be unstable.")
            if config.score:
                scores = subset[config.score].to_numpy()
                labels = truth.to_numpy()
                bins = np.minimum((scores * config.calibration_bins).astype(int), config.calibration_bins - 1)
                ece = 0.0
                for b in np.unique(bins):
                    mask = bins == b
                    predicted, observed = float(scores[mask].mean()), float(labels[mask].mean())
                    count = int(mask.sum())
                    ece += count / len(subset) * abs(predicted - observed)
                    calibration.append(dict(group=key, bin=int(b), count=count,
                                            mean_score=predicted, observed_rate=observed))
                table.loc[key, 'ece'] = ece
                table.loc[key, 'brier_score'] = float(np.mean((scores - labels) ** 2))
        enough = len(table) >= 2
        if not enough:
            warnings.append('Fewer than two observed groups; disparities cannot be assessed.')
        complete_tpr = enough and table.true_positive_rate.notna().all()
        complete_fpr = enough and table.false_positive_rate.notna().all()
        if not (complete_tpr and complete_fpr):
            warnings.append('Equalized odds needs positive and negative examples in every group; unavailable denominators are not treated as zero.')
        disparities = {
            'demographic_parity_difference': float(demographic_parity_difference(y, p, sensitive_features=keys)) if enough else None,
            'demographic_parity_ratio': float(demographic_parity_ratio(y, p, sensitive_features=keys)) if enough and table.selection_rate.max() > 0 else None,
            'equal_opportunity_difference': _gap(table.true_positive_rate),
            'equalized_odds_difference': float(equalized_odds_difference(y, p, sensitive_features=keys)) if complete_tpr and complete_fpr else None,
            'accuracy_gap': _gap(table.accuracy),
            'precision_gap': _gap(table.precision),
            'recall_gap': _gap(table.recall),
            'f1_gap': _gap(table.f1),
            'ece_gap': _gap(table.ece) if config.score else None,
        }
        if not config.score:
            warnings.append('Calibration unavailable: provide probabilities for outcome 1, not hard labels.')
        assessments.append(dict(attributes=list(columns), groups=table.reset_index(names='group').to_dict('records'),
                                disparities=disparities, calibration=calibration, warnings=warnings))
    result = dict(schema_version='1.0', created_at=datetime.now(timezone.utc).isoformat(),
                  rows=len(frame), config=asdict(config), assessments=assessments)
    from app.services.risk_card import assess_risk
    result['risk'] = assess_risk(result)
    return _clean(result)
