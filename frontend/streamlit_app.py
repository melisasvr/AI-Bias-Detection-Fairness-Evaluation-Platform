"""Interactive local fairness assessment workbench."""
from io import StringIO
from pathlib import Path
import sys
import json
import math
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.schemas import EvaluationConfig
from app.services.metrics import evaluate
from app.services.mitigation import threshold_sweep
from app.services.reporting import render_report, to_json
from app.services.risk_card import render_card

# Presentation helpers leave the underlying metrics and machine-readable exports unchanged.
METRIC_LABELS = {
    'selection_rate': 'Selection rate',
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1 score',
    'true_positive_rate': 'True positive rate',
    'false_positive_rate': 'False positive rate',
    'demographic_parity_difference': 'Demographic parity gap',
    'demographic_parity_ratio': 'Demographic parity ratio',
    'equal_opportunity_difference': 'Equal opportunity gap',
    'equalized_odds_difference': 'Equalized odds gap',
    'accuracy_gap': 'Accuracy gap',
    'precision_gap': 'Precision gap',
    'recall_gap': 'Recall gap',
    'f1_gap': 'F1 score gap',
    'ece_gap': 'Calibration error gap',
    'ece': 'Expected calibration error',
    'brier_score': 'Brier score',
    'count': 'Records', 'positives': 'Positive labels', 'negatives': 'Negative labels',
}
RATE_COLUMNS = {
    'selection_rate', 'accuracy', 'precision', 'recall',
    'true_positive_rate', 'false_positive_rate', 'ece',
}


def display_group(key: str, attributes: list[str]) -> str:
    """Decode internal tuple IDs without changing arbitrary uploaded group names."""
    try:
        values = json.loads(key)
    except (ValueError, TypeError):
        return str(key)
    if not isinstance(values, list) or len(values) != len(attributes):
        return str(key)
    aliases = {'group_A': 'Group A', 'group_B': 'Group B'}
    values = [aliases.get(str(value), str(value)) for value in values]
    if len(attributes) == 1:
        return values[0]
    return ' · '.join(f'{attribute}: {value}' for attribute, value in zip(attributes, values))


def format_value(value: float | None, *, kind: str = 'rate') -> str:
    """Keep ratios and score units distinct from rates and percentage-point gaps."""
    if value is None or not math.isfinite(float(value)):
        return 'Unavailable'
    if kind == 'gap':
        return f'{value * 100:.1f} pp'
    if kind == 'score':
        return f'{value:.3f}'
    if kind == 'ratio':
        return f'{value:.3f}'
    return f'{value:.1%}'


def gap_value(metric: str, value: float | None) -> str:
    if metric == 'demographic_parity_ratio':
        return format_value(value, kind='ratio')
    if metric == 'f1_gap':
        return format_value(value, kind='score') + (' score units' if value is not None else '')
    return format_value(value, kind='gap')


def group_table(raw: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    """Create a display-only table; retain raw values for CSV/JSON exports."""
    table = raw.copy()
    table['group'] = table['group'].map(lambda key: display_group(key, attributes))
    for column in table.columns:
        if column in RATE_COLUMNS:
            table[column] = table[column].map(format_value)
        elif column in {'f1', 'brier_score'}:
            table[column] = table[column].map(lambda value: format_value(value, kind='score'))
        elif column in {'count', 'positives', 'negatives'}:
            table[column] = table[column].astype(int)
    return table.rename(columns={'group': 'Group', **METRIC_LABELS})


def explain_gap(assessment: dict[str, Any], metric: str) -> str:
    """Describe the groups behind a disparity; equalized odds uses its larger component."""
    groups = pd.DataFrame(assessment['groups'])
    mapping = {
        'demographic_parity_difference': 'selection_rate',
        'equal_opportunity_difference': 'true_positive_rate',
        'accuracy_gap': 'accuracy', 'precision_gap': 'precision',
        'recall_gap': 'recall', 'f1_gap': 'f1', 'ece_gap': 'ece',
    }
    if metric == 'equalized_odds_difference':
        options = ['true_positive_rate', 'false_positive_rate']
        if any(groups[column].isna().any() for column in options):
            return 'A group is missing a required denominator.'
        column = max(options, key=lambda name: groups[name].max() - groups[name].min())
    else:
        column = mapping.get(metric)
    if column is None or column not in groups or groups[column].isna().any():
        return 'The required group comparison is unavailable.'
    low = groups.loc[groups[column].idxmin()]
    high = groups.loc[groups[column].idxmax()]
    label = METRIC_LABELS[column].lower()
    kind = 'score' if column == 'f1' else 'rate'
    if low[column] == high[column]:
        return f'All observed groups have the same {label}: {format_value(low[column], kind=kind)}.'
    low_name = display_group(low['group'], assessment['attributes'])
    high_name = display_group(high['group'], assessment['attributes'])
    return (f'{label.capitalize()} is {format_value(low[column], kind=kind)} for {low_name}, '
            f'compared with {format_value(high[column], kind=kind)} for {high_name}.')


st.set_page_config(page_title='Fairness Lab', page_icon='⚖️', layout='wide')
st.title('Fairness Lab')
st.caption('AI BIAS DETECTION & FAIRNESS EVALUATION PLATFORM')
st.write('Inspect group outcomes, understand disparities, and document a responsible model review.')
with st.sidebar:
    st.header('Evaluation setup')
    source = st.radio('Data source', ['Synthetic demo', 'Public diabetes demo', 'Upload CSV'])
    uploaded = st.file_uploader('Upload predictions', type=['csv']) if source == 'Upload CSV' else None
    st.caption('Local processing. Uploads are held in session memory; the app does not intentionally save them.')
    with st.expander('Review policy'):
        minimum = st.number_input('Minimum group records', 1, 10000, 30)
        medium = st.slider('Medium absolute gap', 0.01, 0.49, 0.10, 0.01)
        high = st.slider('High absolute gap', 0.02, 1.00, 0.20, 0.01)
        bins = st.slider('Calibration bins', 2, 20, 10)

if source == 'Upload CSV' and uploaded is None:
    st.info('Upload a CSV with binary ground truth, binary predictions, and group columns. Probabilities are optional.')
    st.code('y_true,y_pred,y_score,gender,age_group\n1,1,0.84,group_A,30-49\n0,0,0.12,group_B,50+')
    st.stop()
try:
    if uploaded is not None:
        if uploaded.size > 20 * 1024 * 1024:
            raise ValueError('Upload limit is 20 MB.')
        # Check the raw header before pandas can rename duplicate columns.
        import csv
        decoded = uploaded.getvalue().decode('utf-8-sig')
        header = next(csv.reader(StringIO(decoded)))
        if len(header) != len(set(header)):
            raise ValueError('CSV column names must be unique.')
        data = pd.read_csv(StringIO(decoded), nrows=200001)
    else:
        name = 'synthetic.csv' if source == 'Synthetic demo' else 'diabetes_test.csv'
        data = pd.read_csv(ROOT / 'data' / name)
except (ValueError, UnicodeError, OSError, pd.errors.ParserError, StopIteration) as exc:
    st.error(f'Cannot read dataset: {exc}')
    st.stop()

if source == 'Public diabetes demo':
    st.info('Public-data teaching example: disease progression is converted to a binary target using the training median. Sex labels retain dataset encoding; age bands use a training median split. This is not a clinical model or clinical fairness assessment.')
columns = data.columns.tolist()
def default_index(name: str) -> int:
    return columns.index(name) if name in columns else 0

with st.sidebar:
    target = st.selectbox('Ground truth (0/1)', columns, index=default_index('y_true'))
    prediction = st.selectbox('Predicted label (0/1)', columns, index=default_index('y_pred'))
    score_options = ['None', *columns]
    score = st.selectbox('Probability of outcome 1', score_options, index=score_options.index('y_score') if 'y_score' in columns else 0)
    defaults = [c for c in ('gender', 'age_group', 'sex_code', 'age_band') if c in columns]
    sensitive = st.multiselect('Sensitive / audit attributes', columns, default=defaults)
    st.caption('Outcome 1 must have the same meaning in labels, predictions, and probabilities. Differences are absolute proportions.')

with st.expander('Input preview'):
    st.dataframe(data.head(20), hide_index=True, width='stretch')
    st.write(f'{len(data):,} records · {len(columns)} columns')
try:
    config = EvaluationConfig(target, prediction, tuple(sensitive), None if score == 'None' else score,
                              int(minimum), medium, high, bins)
    report = evaluate(data, config)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

risk = report['risk']
a, b, c = st.columns(3)
a.metric('Review risk', risk['level'])
b.metric('Records evaluated', f"{report['rows']:,}")
largest = risk['findings'][0] if risk['findings'] else None
c.metric('Largest monitored gap', gap_value(largest['metric'], largest['gap']) if largest else 'Unavailable')
if largest:
    largest_view = next(item for item in report['assessments'] if item['attributes'] == largest['attributes'])
    st.caption(
        f"Largest gap across all selected comparisons: {METRIC_LABELS[largest['metric']]} "
        f"for {' × '.join(largest['attributes'])}. "
        + explain_gap(largest_view, largest['metric'])
    )
status_message = {
    'High': 'High review risk — at least one monitored gap reaches the High threshold.',
    'Medium': 'Medium review risk — a gap reaches the Medium threshold, or group evidence is sparse or incomplete.',
    'Low': 'Low review risk — observed gaps are below the configured thresholds with adequate label-based comparisons.',
}
{'High': st.error, 'Medium': st.warning, 'Low': st.success}[risk['level']](status_message[risk['level']])
st.caption('Risk is a configurable screening heuristic. It does not establish discrimination, causality, or compliance.')
selected = st.selectbox('Group comparison', range(len(report['assessments'])),
                        format_func=lambda i: ' × '.join(report['assessments'][i]['attributes']))
view = report['assessments'][selected]
groups = pd.DataFrame(view['groups'])
chart_groups = groups.copy()
chart_groups['group_label'] = chart_groups['group'].map(lambda key: display_group(key, view['attributes']))
for warning in view['warnings']:
    st.warning(warning)

overview, calibration_tab, mitigation_tab, exports = st.tabs(['Group performance', 'Calibration', 'Threshold exploration', 'Reports & Risk Card'])
with overview:
    st.subheader('What the results show')
    for name in ('demographic_parity_difference', 'equal_opportunity_difference'):
        value = view['disparities'][name]
        if value is not None:
            st.write(f'{METRIC_LABELS[name]}: {gap_value(name, value)}. ' + explain_gap(view, name))
    metric = st.selectbox('Compare metric', ['selection_rate', 'accuracy', 'precision', 'recall', 'f1', 'true_positive_rate', 'false_positive_rate'], format_func=lambda name: METRIC_LABELS[name])
    fig = px.bar(chart_groups, x='group_label', y=metric, color='group_label', hover_data=['count', 'positives', 'negatives'],
                 labels={'group_label': 'Observed group', metric: METRIC_LABELS[metric]},
                 color_discrete_sequence=['#4267A5', '#B86A49', '#578B7E', '#8C79AA'], template='plotly_white')
    fig.update_yaxes(tickformat='.2f' if metric == 'f1' else '.0%')
    fig.update_traces(hovertemplate='%{x}<br>' + METRIC_LABELS[metric] + (': %{y:.3f}' if metric == 'f1' else ': %{y:.1%}') + '<br>Records: %{customdata[0]:.0f}<br>Positive labels: %{customdata[1]:.0f}<br>Negative labels: %{customdata[2]:.0f}<extra></extra>')
    fig.update_layout(showlegend=False, yaxis_range=[0, 1], height=430)
    st.plotly_chart(fig, width='stretch')
    st.caption('Undefined values appear as gaps. Hover to inspect group support. Charts show point estimates without confidence intervals.')
    st.subheader('Disparity summary')
    st.dataframe(pd.DataFrame([{'Metric': METRIC_LABELS[k], 'Value': gap_value(k, v)} for k, v in view['disparities'].items()]), hide_index=True, width='stretch')
    st.caption('pp = percentage points. Parity ratio is a unitless ratio (ideal: 1). F1 is shown on its 0–1 score scale. Undefined values are marked Unavailable.')
    st.subheader('Per-group metrics')
    st.dataframe(group_table(groups, view['attributes']), hide_index=True, width='stretch')
with calibration_tab:
    if not view['calibration']:
        st.info('Choose a probability column to assess calibration. Hard labels cannot measure probability calibration.')
    else:
        points = pd.DataFrame(view['calibration'])
        points['group'] = points['group'].map(lambda key: display_group(key, view['attributes']))
        fig = px.line(points, x='mean_score', y='observed_rate', color='group', markers=True,
                      hover_data=['count', 'bin'], template='plotly_white')
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Ideal calibration', line=dict(dash='dash', color='gray')))
        fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1], xaxis_title='Mean predicted probability', yaxis_title='Observed positive rate')
        fig.update_xaxes(tickformat='.0%')
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, width='stretch')
        st.dataframe(group_table(groups[['group', 'count', 'ece', 'brier_score']], view['attributes']), hide_index=True, width='stretch')
        st.caption('ECE: count-weighted absolute calibration error across equal-width probability bins. ECE depends on binning and sample size; Brier score also measures predictive quality.')
with mitigation_tab:
    st.write('Explore how one shared decision threshold changes accuracy and the worst equalized-odds gap across all selected views.')
    st.caption('Exploration uses the current records. Repeated comparisons on test data cause selection leakage. This feature does not optimize or deploy a policy.')
    acknowledged = st.checkbox('I am using validation data for this exploration.')
    if st.button('Compare threshold candidates', disabled=not config.score or not acknowledged):
        with st.spinner('Evaluating 19 thresholds…'):
            sweep = threshold_sweep(data, config)
        plot_data = sweep.assign(gap_pp=sweep['worst_equalized_odds_gap'] * 100)
        fig = px.scatter(plot_data, x='gap_pp', y='accuracy', color='threshold', hover_data=['threshold'],
                         labels={'gap_pp': 'Worst equalized odds gap (percentage points)', 'accuracy': 'Accuracy'}, template='plotly_white')
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, width='stretch')
        display_sweep = pd.DataFrame({
            'Threshold': sweep['threshold'].map(lambda value: f'{value:.2f}'),
            'Accuracy': sweep['accuracy'].map(format_value),
            'Worst equalized odds gap': sweep['worst_equalized_odds_gap'].map(lambda value: format_value(value, kind='gap')),
        })
        st.dataframe(display_sweep, hide_index=True, width='stretch')
        st.download_button('Download threshold comparison', sweep.to_csv(index=False), 'threshold-comparison.csv', 'text/csv')
    st.write('Other review options: improve group coverage, audit labels, compare reweighting or Fairlearn reductions, and test score calibration on separate data. Group-specific policies need domain and governance review.')
with exports:
    st.markdown(render_card(report))
    left, middle, right = st.columns(3)
    left.download_button('Risk Card · Markdown', render_card(report), 'fairness-risk-card.md', 'text/markdown')
    middle.download_button('Full report · Markdown', render_report(report), 'fairness-report.md', 'text/markdown')
    right.download_button('Structured report · JSON', to_json(report), 'fairness-report.json', 'application/json')
    st.download_button('Current group table · CSV', groups.to_csv(index=False), 'group-metrics.csv', 'text/csv')
