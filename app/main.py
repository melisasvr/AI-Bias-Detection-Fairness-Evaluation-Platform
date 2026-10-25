"""CLI for reproducible evaluation without the interface."""
import argparse
from pathlib import Path
import pandas as pd
from app.schemas import EvaluationConfig
from app.services.metrics import evaluate
from app.services.reporting import render_report, to_json
from app.services.risk_card import render_card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--target', default='y_true')
    parser.add_argument('--prediction', default='y_pred')
    parser.add_argument('--sensitive', nargs='+', required=True)
    parser.add_argument('--score')
    parser.add_argument('--output', type=Path, default=Path('reports'))
    args = parser.parse_args()
    try:
        report = evaluate(pd.read_csv(args.csv), EvaluationConfig(args.target, args.prediction, tuple(args.sensitive), args.score))
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in [('fairness-report.md', render_report(report)), ('fairness-risk-card.md', render_card(report)), ('fairness-report.json', to_json(report))]:
        (args.output / name).write_text(content, encoding='utf-8')
    print(f"Risk: {report['risk']['level']}. Reports written to {args.output.resolve()}")

if __name__ == '__main__':
    main()
