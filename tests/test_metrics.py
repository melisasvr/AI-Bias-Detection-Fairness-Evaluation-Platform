import json
import numpy as np
import pandas as pd
import pytest
from app.schemas import EvaluationConfig
from app.services.metrics import evaluate
from app.services.reporting import render_report, to_json
from app.services.risk_card import render_card
from app.services.mitigation import threshold_sweep

@pytest.fixture
def sample():
    # A: TP=2, FP=1, TN=1, FN=0. B: TP=1, FP=0, TN=2, FN=1.
    return pd.DataFrame({'y_true': [1,1,0,0,1,1,0,0],
                         'y_pred': [1,1,1,0,1,0,0,0],
                         'group': ['A']*4 + ['B']*4,
                         'score': [0.9,0.8,0.7,0.1,0.6,0.4,0.2,0.1]})

def config(**kwargs):
    return EvaluationConfig('y_true', 'y_pred', ('group',), min_group_size=1, **kwargs)

def test_hand_calculated_metrics(sample):
    report = evaluate(sample, config())
    d = report['assessments'][0]['disparities']
    assert d['demographic_parity_difference'] == pytest.approx(0.5)
    assert d['demographic_parity_ratio'] == pytest.approx(1/3)
    assert d['equal_opportunity_difference'] == pytest.approx(0.5)
    assert d['equalized_odds_difference'] == pytest.approx(0.5)
    g = report['assessments'][0]['groups'][0]
    assert g['accuracy'] == pytest.approx(0.75)
    assert g['precision'] == pytest.approx(2/3)
    assert g['recall'] == 1
    assert g['f1'] == pytest.approx(0.8)
    assert report['risk']['level'] == 'High'

def test_calibration(sample):
    report = evaluate(sample, config(score='score', calibration_bins=2))
    a = report['assessments'][0]['groups'][0]
    assert a['ece'] == pytest.approx(0.125)
    assert a['brier_score'] == pytest.approx(0.1375)
    assert sum(b['count'] for b in report['assessments'][0]['calibration']) == len(sample)

def test_undefined_denominators(sample):
    sample.loc[sample.group == 'B', 'y_true'] = 0
    report = evaluate(sample, config())
    d = report['assessments'][0]['disparities']
    assert d['equal_opportunity_difference'] is None
    assert d['equalized_odds_difference'] is None
    assert report['risk']['incomplete']
    json.loads(to_json(report))

def test_all_negative_predictions(sample):
    sample.y_pred = 0
    r = evaluate(sample, config())
    assert r['assessments'][0]['disparities']['demographic_parity_ratio'] is None
    assert r['assessments'][0]['groups'][0]['precision'] is None
    assert r['risk']['level'] != 'Low'

def test_single_group(sample):
    sample.group = 'A'
    r = evaluate(sample, config())
    assert r['assessments'][0]['disparities']['demographic_parity_difference'] is None
    assert r['risk']['level'] == 'Medium'

def test_intersections_and_delimiter_collision(sample):
    sample['other'] = ['x|y','x|y','z','z','x|y','x|y','z','z']
    r = evaluate(sample, EvaluationConfig('y_true','y_pred',('group','other'),min_group_size=1))
    assert len(r['assessments']) == 3
    assert len(r['assessments'][2]['groups']) == 4

@pytest.mark.parametrize('column,value', [('y_true',2),('y_pred',None),('score',np.inf),('score',-0.1),('group','')])
def test_invalid_input(sample,column,value):
    sample.loc[0,column] = value
    with pytest.raises(ValueError):
        evaluate(sample, config(score='score'))

def test_sparse_cannot_be_low(sample):
    sample.y_pred = sample.y_true
    assert evaluate(sample, EvaluationConfig('y_true','y_pred',('group',),min_group_size=30))['risk']['level'] == 'Medium'

def test_reports_and_thresholds(sample):
    r = evaluate(sample, config(score='score'))
    assert 'Fairness Risk Card' in render_card(r)
    assert 'Detailed Fairness Report' in render_report(r)
    assert json.loads(to_json(r))['schema_version'] == '1.0'
    sweep = threshold_sweep(sample, config(score='score'))
    assert len(sweep) == 19
    assert sweep.accuracy.between(0,1).all()
    assert (sample.y_pred == [1,1,1,0,1,0,0,0]).all()
