# Fairness Risk Card

**Screening risk: High**

Evaluated 89 records at 2026-10-25T11:48:29.890887+00:00.
Outcome 1: y_true; prediction: y_pred.
Attributes: sex_code, age_band.

## Key disparities (maximum minus minimum)

- sex_code × age_band: equalized_odds_difference = 0.433
- sex_code × age_band: demographic_parity_difference = 0.321
- age_band: equalized_odds_difference = 0.314
- sex_code × age_band: precision_gap = 0.246
- sex_code × age_band: equal_opportunity_difference = 0.208

## Review policy

Medium ≥ 0.10; High ≥ 0.20 on any monitored absolute gap (including ECE gap).
Groups below 30 records or undefined label-based gaps raise a Low screen to Medium.
Missing probabilities leave calibration unassessed. The parity ratio is reported separately, not used in this risk heuristic.

## Recommendations

- Review group sample sizes, label quality, and how outcome 1 relates to benefit or harm.
- Investigate the largest gaps with domain stakeholders before selecting a fairness objective.
- Compare threshold candidates on validation data, then audit once on an untouched test set.
- Consider better data coverage, reweighting, or Fairlearn mitigation; compare utility trade-offs.

## Limitations

Descriptive point estimates only; no confidence intervals or causal attribution. Low means lower observed gaps under this policy, not proof of fairness.
No legal compliance claim. Fairness objectives may conflict. Labels and selected groups can reflect historical bias.
No individual records are included; aggregate group labels can still be sensitive. Review exports before sharing.

## Data quality notes

- sex_code × age_band: ["encoded_1", "at_or_above_training_median"]: only 21 records; estimates may be unstable.
- sex_code × age_band: ["encoded_1", "below_training_median"]: only 26 records; estimates may be unstable.
- sex_code × age_band: ["encoded_2", "at_or_above_training_median"]: only 25 records; estimates may be unstable.
- sex_code × age_band: ["encoded_2", "below_training_median"]: only 17 records; estimates may be unstable.

# Detailed Fairness Report

## sex_code

| Metric | Value |
| --- | ---: |
| demographic_parity_difference | 0.1059 |
| demographic_parity_ratio | 0.8221 |
| equal_opportunity_difference | 0.0326 |
| equalized_odds_difference | 0.1510 |
| accuracy_gap | 0.0517 |
| precision_gap | 0.0626 |
| recall_gap | 0.0326 |
| f1_gap | 0.0160 |
| ece_gap | 0.0583 |

Per-group metrics (JSON; null = undefined):

```json
[
  {
    "group": "[\"encoded_1\"]",
    "accuracy": 0.7659574468085106,
    "precision": 0.782608695652174,
    "recall": 0.75,
    "f1": 0.7659574468085106,
    "selection_rate": 0.48936170212765956,
    "true_positive_rate": 0.75,
    "false_positive_rate": 0.21739130434782608,
    "count": 47.0,
    "positives": 24.0,
    "negatives": 23.0,
    "ece": 0.13099886727681395,
    "brier_score": 0.14118610619721145
  },
  {
    "group": "[\"encoded_2\"]",
    "accuracy": 0.7142857142857143,
    "precision": 0.72,
    "recall": 0.782608695652174,
    "f1": 0.75,
    "selection_rate": 0.5952380952380952,
    "true_positive_rate": 0.782608695652174,
    "false_positive_rate": 0.3684210526315789,
    "count": 42.0,
    "positives": 23.0,
    "negatives": 19.0,
    "ece": 0.18930139979331756,
    "brier_score": 0.16604769673638234
  }
]
```

## age_band

| Metric | Value |
| --- | ---: |
| demographic_parity_difference | 0.1436 |
| demographic_parity_ratio | 0.7641 |
| equal_opportunity_difference | 0.0788 |
| equalized_odds_difference | 0.3136 |
| accuracy_gap | 0.1850 |
| precision_gap | 0.1714 |
| recall_gap | 0.0788 |
| f1_gap | 0.1256 |
| ece_gap | 0.0210 |

Per-group metrics (JSON; null = undefined):

```json
[
  {
    "group": "[\"at_or_above_training_median\"]",
    "accuracy": 0.6521739130434783,
    "precision": 0.6785714285714286,
    "recall": 0.7307692307692307,
    "f1": 0.7037037037037037,
    "selection_rate": 0.6086956521739131,
    "true_positive_rate": 0.7307692307692307,
    "false_positive_rate": 0.45,
    "count": 46.0,
    "positives": 26.0,
    "negatives": 20.0,
    "ece": 0.19266632076869974,
    "brier_score": 0.18171744101146747
  },
  {
    "group": "[\"below_training_median\"]",
    "accuracy": 0.8372093023255814,
    "precision": 0.85,
    "recall": 0.8095238095238095,
    "f1": 0.8292682926829268,
    "selection_rate": 0.46511627906976744,
    "true_positive_rate": 0.8095238095238095,
    "false_positive_rate": 0.13636363636363635,
    "count": 43.0,
    "positives": 21.0,
    "negatives": 22.0,
    "ece": 0.17167098394668523,
    "brier_score": 0.12211041785277892
  }
]
```

## sex_code × age_band

| Metric | Value |
| --- | ---: |
| demographic_parity_difference | 0.3205 |
| demographic_parity_ratio | 0.5192 |
| equal_opportunity_difference | 0.2077 |
| equalized_odds_difference | 0.4333 |
| accuracy_gap | 0.2062 |
| precision_gap | 0.2460 |
| recall_gap | 0.2077 |
| f1_gap | 0.1905 |
| ece_gap | 0.1175 |

Per-group metrics (JSON; null = undefined):

```json
[
  {
    "group": "[\"encoded_1\", \"at_or_above_training_median\"]",
    "accuracy": 0.6666666666666666,
    "precision": 0.7142857142857143,
    "recall": 0.7692307692307693,
    "f1": 0.7407407407407407,
    "selection_rate": 0.6666666666666666,
    "true_positive_rate": 0.7692307692307693,
    "false_positive_rate": 0.5,
    "count": 21.0,
    "positives": 13.0,
    "negatives": 8.0,
    "ece": 0.19376411398478885,
    "brier_score": 0.19127010641736497
  },
  {
    "group": "[\"encoded_1\", \"below_training_median\"]",
    "accuracy": 0.8461538461538461,
    "precision": 0.8888888888888888,
    "recall": 0.7272727272727273,
    "f1": 0.8,
    "selection_rate": 0.34615384615384615,
    "true_positive_rate": 0.7272727272727273,
    "false_positive_rate": 0.06666666666666667,
    "count": 26.0,
    "positives": 11.0,
    "negatives": 15.0,
    "ece": 0.14972748785000162,
    "brier_score": 0.10073364448093361
  },
  {
    "group": "[\"encoded_2\", \"at_or_above_training_median\"]",
    "accuracy": 0.64,
    "precision": 0.6428571428571429,
    "recall": 0.6923076923076923,
    "f1": 0.6666666666666666,
    "selection_rate": 0.56,
    "true_positive_rate": 0.6923076923076923,
    "false_positive_rate": 0.4166666666666667,
    "count": 25.0,
    "positives": 13.0,
    "negatives": 12.0,
    "ece": 0.2672420859509353,
    "brier_score": 0.17369320207051359
  },
  {
    "group": "[\"encoded_2\", \"below_training_median\"]",
    "accuracy": 0.8235294117647058,
    "precision": 0.8181818181818182,
    "recall": 0.9,
    "f1": 0.8571428571428571,
    "selection_rate": 0.6470588235294118,
    "true_positive_rate": 0.9,
    "false_positive_rate": 0.2857142857142857,
    "count": 17.0,
    "positives": 10.0,
    "negatives": 7.0,
    "ece": 0.22563569309628442,
    "brier_score": 0.15480430653913052
  }
]
```

