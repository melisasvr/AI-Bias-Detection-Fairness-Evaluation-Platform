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
