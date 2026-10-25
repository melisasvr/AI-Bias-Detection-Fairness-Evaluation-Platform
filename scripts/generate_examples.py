"""Generate deterministic synthetic and public-data examples without downloads."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    destination = ROOT / 'data'
    destination.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    n = 1200
    gender = rng.choice(['group_A', 'group_B'], n)
    age = rng.choice(['under_40', '40_plus'], n)
    latent = rng.normal(size=n)
    probability = 1 / (1 + np.exp(-latent))
    truth = rng.binomial(1, probability)
    # Deliberately introduce a group-related score shift for demonstration.
    score = 1 / (1 + np.exp(-(latent + np.where(gender == 'group_A', 0.65, -0.65))))
    pd.DataFrame(dict(y_true=truth, y_pred=(score >= 0.5).astype(int), y_score=score,
                      gender=gender, age_group=age)).to_csv(destination / 'synthetic.csv', index=False)

    dataset = load_diabetes(as_frame=True)
    features, outcome = dataset.data, dataset.target
    train_idx, remainder = train_test_split(np.arange(len(features)), test_size=0.4, random_state=42)
    validation_idx, test_idx = train_test_split(remainder, test_size=0.5, random_state=43)
    target_cutoff = float(outcome.iloc[train_idx].median())
    age_cutoff = float(features.age.iloc[train_idx].median())
    y = (outcome >= target_cutoff).astype(int)
    # Audit attributes are excluded from the demo classifier; proxies can remain.
    x = features.drop(columns=['sex', 'age'])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    codes = sorted(features.sex.unique())
    for split, indices in [('validation', validation_idx), ('test', test_idx)]:
        scores = model.predict_proba(x.iloc[indices])[:, 1]
        pd.DataFrame(dict(y_true=y.iloc[indices].to_numpy(), y_pred=(scores >= 0.5).astype(int), y_score=scores,
                          sex_code=np.where(features.sex.iloc[indices] == codes[0], 'encoded_1', 'encoded_2'),
                          age_band=np.where(features.age.iloc[indices] < age_cutoff, 'below_training_median', 'at_or_above_training_median'))).to_csv(destination / f'diabetes_{split}.csv', index=False)
    metadata = dict(seed=42, train_rows=len(train_idx), validation_rows=len(validation_idx), test_rows=len(test_idx),
                    target_rule='1 if progression >= training target median', target_cutoff=target_cutoff,
                    age_cutoff_scaled=age_cutoff, sex_codes_scaled=[float(c) for c in codes],
                    predictor_columns=x.columns.tolist(), source='sklearn.datasets.load_diabetes')
    (destination / 'example_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('Generated synthetic.csv, diabetes_validation.csv, diabetes_test.csv, and metadata.')

if __name__ == '__main__':
    main()
