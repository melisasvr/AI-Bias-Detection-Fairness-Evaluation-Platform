"""Transparent screening heuristics, not compliance or causal conclusions."""
from typing import Any
import math

RISK_METRICS = ('demographic_parity_difference', 'equal_opportunity_difference',
                'equalized_odds_difference', 'accuracy_gap', 'precision_gap',
                'recall_gap', 'f1_gap', 'ece_gap')


def assess_risk(report: dict[str, Any]) -> dict[str, Any]:
    config = report['config']
    findings = []
    incomplete = False
    sparse = False
    for view in report['assessments']:
        for name in RISK_METRICS:
            value = view['disparities'][name]
            if value is None or not math.isfinite(value):
                if name != 'ece_gap' or config['score']:
                    incomplete = True
                continue
            findings.append(dict(attributes=view['attributes'], metric=name, gap=value))
        sparse |= any(g['count'] < config['min_group_size'] for g in view['groups'])
    findings.sort(key=lambda item: item['gap'], reverse=True)
    worst = findings[0]['gap'] if findings else None
    level = 'High' if worst is not None and worst >= config['high_gap'] else (
        'Medium' if worst is not None and worst >= config['medium_gap'] else 'Low')
    if incomplete or sparse:
        if level == 'Low':
            level = 'Medium'
    return dict(level=level, worst_gap=worst, incomplete=incomplete, sparse=sparse,
                findings=findings[:5], recommendations=[
                    'Review group sample sizes, label quality, and how outcome 1 relates to benefit or harm.',
                    'Investigate the largest gaps with domain stakeholders before selecting a fairness objective.',
                    'Compare threshold candidates on validation data, then audit once on an untouched test set.',
                    'Consider better data coverage, reweighting, or Fairlearn mitigation; compare utility trade-offs.',
                ])


def _safe(text: object) -> str:
    return str(text).replace('\n', ' ').replace('|', r'\|').replace('<', '&lt;').replace('>', '&gt;')


def render_card(report: dict[str, Any]) -> str:
    risk = report['risk']
    config = report['config']
    lines = ['# Fairness Risk Card', '', f"**Screening risk: {risk['level']}**", '',
             f"Evaluated {report['rows']:,} records at {report['created_at']}.",
             f"Outcome 1: {_safe(config['target'])}; prediction: {_safe(config['prediction'])}.",
             f"Attributes: {_safe(', '.join(config['sensitive']))}.", '',
             '## Key disparities (maximum minus minimum)', '']
    for item in risk['findings']:
        lines.append(f"- {_safe(' × '.join(item['attributes']))}: {item['metric']} = {item['gap']:.3f}")
    if not risk['findings']:
        lines.append('- Insufficient comparable groups.')
    lines += ['', '## Review policy', '',
              f"Medium ≥ {config['medium_gap']:.2f}; High ≥ {config['high_gap']:.2f} on any monitored absolute gap (including ECE gap).",
              f"Groups below {config['min_group_size']} records or undefined label-based gaps raise a Low screen to Medium.",
              'Missing probabilities leave calibration unassessed. The parity ratio is reported separately, not used in this risk heuristic.',
              '', '## Recommendations', '']
    lines.extend(f'- {item}' for item in risk['recommendations'])
    lines += ['', '## Limitations', '',
              'Descriptive point estimates only; no confidence intervals or causal attribution. Low means lower observed gaps under this policy, not proof of fairness.',
              'No legal compliance claim. Fairness objectives may conflict. Labels and selected groups can reflect historical bias.',
              'No individual records are included; aggregate group labels can still be sensitive. Review exports before sharing.',
              '', '## Data quality notes', '']
    for view in report['assessments']:
        lines.extend(f"- {_safe(' × '.join(view['attributes']))}: {_safe(w)}" for w in view['warnings'])
    return '\n'.join(lines) + '\n'
