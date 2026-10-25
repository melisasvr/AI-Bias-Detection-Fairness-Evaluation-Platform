"""Portable report exports; JSON null represents undefined metrics."""
import json
from typing import Any
from app.services.risk_card import render_card


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)


def render_report(report: dict[str, Any]) -> str:
    text = render_card(report) + '\n# Detailed Fairness Report\n\n'
    for view in report['assessments']:
        text += '## ' + ' × '.join(view['attributes']).replace('<', '&lt;') + '\n\n'
        text += '| Metric | Value |\n| --- | ---: |\n'
        for name, value in view['disparities'].items():
            text += f"| {name} | {value:.4f} |\n" if value is not None else f'| {name} | Undefined / unavailable |\n'
        text += '\nPer-group metrics (JSON; null = undefined):\n\n'
        text += '```json\n' + json.dumps(view['groups'], indent=2, ensure_ascii=True).replace('```', '\\u0060' * 3) + '\n```\n\n'
    return text
