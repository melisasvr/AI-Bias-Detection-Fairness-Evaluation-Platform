# AI Bias Detection & Fairness Evaluation Platform

A local Python workbench for auditing binary classifier predictions across groups.
Upload a CSV, compare group outcomes, inspect calibration, explore threshold trade-offs,
and export a Fairness Report and a model-card-ready Fairness Risk Card.

Built with **Fairlearn, scikit-learn, pandas, NumPy, Streamlit, and Plotly**.
Pure Streamlit keeps the first version easy to install; the service layer is reusable
from scripts or a future FastAPI backend.

## Features

- Upload a CSV and configure label, prediction, probability, and audit columns.
- Assess individual attributes and observed intersections with Fairlearn.
- Compare group accuracy, precision, recall, F1, selection rates, TPR, and FPR.
- Inspect group calibration curves, expected calibration error, and Brier scores.
- Read plain-language disparity findings and identify the comparison behind the largest gap.
- Review a color-coded, configurable Low/Medium/High screening result.
- Explore shared-threshold trade-offs on validation data.
- Download a Markdown Fairness Report and Risk Card, structured JSON, and CSV tables.

## Example findings

The included synthetic demo contains **1,200 fictional records** and deliberately
introduces group-related score shifts. With the default configuration:

| Finding | Result |
| --- | --- |
| Group A selection rate | 72.6% |
| Group B selection rate | 23.8% |
| Demographic parity gap across gender groups | 48.8 percentage points |
| Equal opportunity gap across gender groups | 57.1 percentage points |
| Largest monitored gap across all selected views | 60.8 percentage points |
| Screening risk | High |

The largest gap is the equal opportunity gap in the **gender × age_group**
intersectional view. It differs from the gender-only gap because it compares more
specific groups. Gaps are calculated before rounding the displayed rates.
These results describe the demo and should not be interpreted as findings about
real demographic populations.

## Quick start — Windows PowerShell

Requires Python 3.11 or newer. Clone or download this repository, then open a
terminal in its root folder (the folder containing `requirements.txt`).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c constraints-tested.txt
python -m streamlit run frontend/streamlit_app.py
```

If you have a newer Python instead of 3.11, use `py -m venv .venv`.
If PowerShell prevents activation, call the environment directly; no policy change is needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-tested.txt
.\.venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py
```

Open **http://localhost:8501**. The synthetic demo opens immediately.
The public diabetes demo also works offline after dependency installation.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-tested.txt
python -m streamlit run frontend/streamlit_app.py
```

`requirements.txt` declares supported ranges; `constraints-tested.txt` captures the
direct dependency versions used for this delivery. It is not a full transitive lockfile.

## First evaluation

1. Start with the synthetic demo to see deliberately introduced group score shifts.
2. Choose **Upload CSV**, then map ground truth, predictions, optional probabilities,
   and one or more sensitive/audit attributes in the sidebar.
3. Select a group comparison. Each attribute is evaluated on its own; selecting
   multiple attributes adds a combined intersectional view of observed groups.
4. Inspect group sizes and undefined values alongside disparities. Open the
   calibration tab when probabilities are available.
5. Download the Markdown Risk Card, detailed Markdown report, structured JSON,
   or the selected per-group CSV table. Attach the Risk Card to a model card.

The current interface evaluates immediately when mappings or review settings change.

## CSV contract

```csv
y_true,y_pred,y_score,gender,age_group
1,1,0.91,group_A,under_40
0,0,0.22,group_B,40_plus
1,0,0.43,group_B,under_40
0,1,0.68,group_A,40_plus
```

| Field | Required | Meaning |
| --- | --- | --- |
| Ground truth | Yes | Binary numeric 0/1 label |
| Predictions | Yes | Binary numeric 0/1 decision |
| Sensitive attributes | Yes, at least one | Nonmissing categorical audit groups |
| Probability | No | Finite probability in [0,1] of **outcome 1** |

Column names are configurable. Map other label encodings to 0/1 before uploading.
Choose the positive outcome deliberately: selection for a benefit and prediction of
an adverse event have different interpretations. Predictions and scores must refer
to the same outcome; scores are not required to use a 0.5 decision threshold.

Missing values in selected columns are rejected; unused columns are ignored.
No records are silently dropped. Continuous sensitive values should be binned before
uploading. Limits: 20 MB upload, 200,000 rows, 100 observed intersection groups,
and 120 characters per group label. All selected roles require distinct columns.
Group IDs are encoded as JSON tuples to avoid delimiter collisions.

## Metrics and definitions

Fairlearn `MetricFrame` provides group-level metrics. Demographic parity and
complete equalized odds assessments call Fairlearn's predefined functions.
Equal opportunity uses the max-minus-min gap of valid true positive rates.

| Metric | Definition | Ideal parity |
| --- | --- | --- |
| Demographic parity difference | max selection rate − min selection rate | 0 |
| Demographic parity ratio | min selection rate / max selection rate | 1 |
| Equal opportunity difference | max TPR − min TPR | 0 |
| Equalized odds difference | max(TPR gap, FPR gap) | 0 |
| Accuracy, precision, recall, F1 | Standard binary classification metrics per group | Context dependent |
| ECE per group | Weighted mean of absolute bin probability/outcome differences | 0 |
| Brier score per group | Mean squared probability error | 0 |

The dashboard displays rates as percentages and rate gaps as **percentage points
(pp)**: a raw gap of 0.10 means **10 percentage points**. F1 and Brier scores stay
on their 0–1 scales, and the demographic parity ratio stays unitless. JSON and CSV
exports retain raw numeric values; Markdown reports retain the service-level values.
Readable group names appear in charts and comparison tables; internal JSON tuple IDs
remain in structured exports for reproducibility.
Differences compare the most separated groups, with no privileged reference group.
Calibration uses equal-width bins (10 by default); the score 1.0 belongs to the final
bin. Reliability diagrams expose the underlying bin counts. ECE varies with binning
and sample size; Brier score is not a pure calibration metric.

Undefined metrics remain `null` in JSON and appear as **Unavailable** in formatted
dashboard tables (or empty numeric cells in raw exports): no positive ground truth
means TPR/recall is undefined; no negative truth means FPR is undefined; no predicted
positives means precision is undefined. F1 is undefined only if its own denominator
is zero. A disparity requiring an undefined group metric is unavailable for the
whole comparison, rather than excluding that group. No group has been assigned a
zero rate solely because its denominator is missing. All-zero selection makes the
parity ratio undefined. Fewer than two groups makes disparities unavailable.

## Risk policy and interpretation

The Risk Card is an explicitly **heuristic screening policy**, not a certification.
The default policy monitors DP, equal opportunity, equalized odds, accuracy, precision,
recall, F1, and ECE absolute gaps across every view:

- **High:** any monitored available gap ≥ 0.20.
- **Medium:** any gap ≥ 0.10, or evidence is incomplete, or any group has fewer than 30 records.
- **Low:** available gaps below 0.10 with adequate counts and complete label-based comparisons.

Users can change gap thresholds, minimum group count, and calibration bins.
The DP ratio is descriptive and does not enter this risk rule. Missing probability
scores leave calibration unassessed, with an explicit warning, and do not alone raise
the risk level. Missing label-based disparities or supplied-score ECE comparisons do.
The report records the settings and UTC generation time for reproducibility.

Equal ECE across groups does not mean either group is well calibrated. Always inspect
absolute errors too. Risk is comparative and sample-specific; a Low screen can still
hide serious harms. There are no confidence intervals, causal claims, or legal tests.
Demographic parity, equalized odds, and calibration can conflict. Label bias and
unmeasured groups can invalidate reassuring results. Review outcomes with affected
stakeholders and domain experts.

## Public dataset example — fully runnable, offline

`data/diabetes_test.csv` and `data/diabetes_validation.csv` contain predictions from
a logistic regression trained on scikit-learn's bundled public diabetes dataset.
The original dataset has 442 records and a continuous disease-progression response.
This project creates a **teaching-only binary task** by thresholding that response at
the training-set median. It is not an established clinical classification endpoint.

The generator splits row indices into training (60%), validation (20%), and test (20%)
before learning the target cutoff, age cutoff, or preprocessing parameters. A scaler
and logistic regression are fit on training records only. The audit attributes `age`
and `sex` are excluded from the classifier inputs; correlated proxies may remain.
The original scaled sex values are labeled `encoded_1` and `encoded_2` without assuming
their real-world identity mapping. Age bands are based on the training median of the
scaled age feature, not invented calendar ages. Metadata records the exact cutoffs,
row counts, predictor columns, and seed.

```bash
python scripts/generate_examples.py
python -m app.main data/diabetes_test.csv --sensitive sex_code age_band --score y_score --output reports/public-example
```

A checked-in public-example report is included. Dependency changes may affect fitted
probabilities slightly; the seed and generation script make the procedure reproducible.

Sources and attribution:

- [scikit-learn diabetes loader and dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- [Original dataset source linked by scikit-learn](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)
- Efron, Hastie, Johnstone and Tibshirani (2004), *Least Angle Regression*, Annals of Statistics.
- [Fairlearn fairness assessment documentation](https://fairlearn.org/v0.13/user_guide/assessment/perform_fairness_assessment.html)

The public CSVs are small derived educational examples. Dataset provenance is separate
from this project's code; review upstream terms for your intended redistribution.
Synthetic group labels and observations are fictional and generated with seed 42.

## Threshold exploration and mitigation

Select the probability column and use **validation data**, then acknowledge this in
the Threshold exploration tab. For the public example, upload `diabetes_validation.csv`.
The tool compares 19 **shared** thresholds (0.05 to 0.95) and shows accuracy versus the
worst equalized-odds gap over all selected group views. Undefined comparisons remain
unavailable. Very small support can make the trade-off unreliable.

This is an exploratory sweep, not a fitted Fairlearn `ThresholdOptimizer`, and it
does not automatically choose a policy or alter the original predictions. The service
returns the candidate comparison table. Selecting thresholds on test data would leak
evaluation information. Choose a fairness objective and utility constraints on
validation data, then evaluate the chosen policy once on an untouched test set.
Group-specific thresholds, Fairlearn reductions, reweighting, improved data coverage,
and score calibration are recommendations for a later reviewed experiment.

## Command line and programmatic use

```bash
python -m app.main data/synthetic.csv --sensitive gender age_group --score y_score --output reports/synthetic-example
```

```python
import pandas as pd
from app.schemas import EvaluationConfig
from app.services.metrics import evaluate

result = evaluate(
    pd.read_csv('data/synthetic.csv'),
    EvaluationConfig('y_true', 'y_pred', ('gender', 'age_group'), score='y_score'),
)
print(result['risk']['level'])
```

The CLI writes `fairness-report.md`, `fairness-risk-card.md`, and
`fairness-report.json` to the selected directory. Running again replaces those output
files. The Streamlit application exports via download buttons and does not write input
CSVs to disk. Aggregate exports contain group labels and small counts: review them
before sharing. Deployments need separate access control, retention, and privacy design.

## Project structure

```text
fairness-platform/
├── app/
│   ├── main.py                 # CLI
│   ├── schemas.py              # Typed configuration
│   ├── services/
│   │   ├── metrics.py          # Fairlearn metrics and calibration
│   │   ├── mitigation.py       # Validation threshold comparison
│   │   ├── reporting.py        # Markdown and strict JSON
│   │   └── risk_card.py        # Screening policy and card
│   └── utils/validation.py
├── frontend/streamlit_app.py
├── scripts/generate_examples.py
├── data/                       # Ready-to-use CSVs and metadata
├── notebooks/README.md
├── reports/public-example/     # Generated example exports
├── tests/
├── .streamlit/config.toml
├── requirements.txt
├── constraints-tested.txt
└── README.md
```

## Verification

```bash
python -m pip install -r requirements-dev.txt -c constraints-tested.txt
python -m pytest -q
```

Verification at delivery: **14 tests passed on Python 3.12**, including interactive
Streamlit checks for both demos, group-view switching, threshold comparisons, and
the empty-upload state.

Manual local checks also confirmed CSV upload and report downloads. These manual
checks are separate from the automated test suite.

Tests check hand-calculated metrics, calibration, invalid inputs, undefined denominators,
single groups, intersection handling, strict JSON exports, risk escalation, threshold
comparisons, and Streamlit rendering/interactions. Tests do not require network access.

## MVP scope and next steps

Implemented: binary predictions, marginal/intersectional assessment, group performance,
calibration, clear data-quality notes, interactive charts, threshold exploration, and
Markdown/JSON/CSV exports. PDF export and FastAPI are optional future additions.
Other next steps: bootstrap confidence intervals, user-defined utility constraints,
held-out Fairlearn mitigation experiments, dataset/version fingerprints, model-card
metadata, and a reviewed domain-specific governance policy.
