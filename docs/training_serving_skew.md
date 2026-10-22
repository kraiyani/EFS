# Training-serving skew: a controlled companion experiment

This experiment demonstrates one source-level mismatch in the legacy application.
It does not repair or invoke that application. It never reads `data/`, existing
`artifacts/`, notebooks, customer records, or a network resource.

## Source-level observation

`src/components/data_transformation.py` fits a `MinMaxScaler` and computes
`transformed_data`, but returns selected columns from the raw `df_x`.
`src/components/model_trainer.py` then splits and trains classifiers on that raw
frame. `src/pipeline/predict_pipeline.py` loads a separately saved scaler and
passes transformed features to the saved random forest. Those inputs use different
coordinates from those used to learn tree thresholds.

There are separate leakage concerns: preprocessing and label-dependent feature
selection occur before the legacy split. The experiment deliberately excludes
feature selection and fits **both** experimental scalers only on training rows.
This isolates the serving mismatch from those other issues. No production impact
is measured, and legacy source and model artifacts remain untouched.

## Fixed protocol

The declared seeds are **11, 29, 47**, with 4,000 independent synthetic rows for
each seed. Every seed uses one stratified 75/25 train/holdout split. Six independent
uniform variables are shifted into positive feature ranges, all strictly above 1:
account age 12–180, balance 10,000–100,000, income 20,000–130,000, late-payment days
2–90, payment count 6–66, and utilisation percent 5–95. Names and the arbitrary
logistic label relationship are illustrative; they encode no credit-risk claim.
IDs and labels never enter the feature matrix.

Before measurement, the random forest was fixed at 160 trees, maximum depth 8,
minimum leaf size 4, one worker, and the current seed as `random_state`; remaining
options use scikit-learn 1.7.2 defaults. There is no seed, model, threshold, or
hyperparameter search. Classification uses the forest's standard prediction rule
(the positive class must win its probability vote; an exact 0.5 tie chooses 0).

For each identical, ordered holdout:

1. **raw_baseline** trains a forest on raw training features and predicts raw rows.
2. **scaled_serving** feeds training-only MinMax-scaled rows to that **same** raw
   forest, isolating the coordinate mismatch.
3. **consistent_pipeline** fits a new `Pipeline([MinMaxScaler, RandomForest])` on
   training rows and predicts from raw holdout requests through that pipeline.

The deliberately positive units make the mismatch extreme and inspectable: scaled
holdout values sit near [0, 1], below the raw model's training feature ranges.
They therefore reach the same extreme leaves. This is not a representative estimate
of real-world degradation. Scaling at both fit and predict time generally preserves
tree thresholds under a positive affine transform, subject to numerical details;
a forest can also be valid with raw inputs at both boundaries.

## Reproduce

Use Python 3.11.7 and an isolated environment. The dedicated pinned requirements
include only experiment dependencies and pytest; they do not install the application:

```sh
python -m pip install -r experiments/requirements-training-serving-skew.txt
python -m pytest tests/test_training_serving_skew.py -q
python -m experiments.training_serving_skew
```

The default destination is `results/training_serving_skew/`; `--output PATH`
selects another directory. The runner replaces its named experiment artifacts
there, so use a dedicated results directory. The requirements pin direct packages,
not a complete transitive lockfile; tested runtime versions are also recorded.

## Inspect the evidence

- `metrics.csv` contains nine rows (three paths by three seeds): ROC-AUC, average
  precision, accuracy, positive counts, classification disagreements, and mean/max
  absolute probability differences relative to the raw baseline.
- `predictions.csv` preserves each holdout's row ID, label and all three class-1
  probabilities and predictions. Comparisons are paired within a seed.
- `synthetic_data.csv` preserves every generated row, seed and split assignment.
  All CSV files explicitly use LF endings and round-trip precision.
- `pipeline_seed_11.joblib`, `pipeline_seed_29.joblib`, and
  `pipeline_seed_47.joblib` contain each newly fitted complete pipeline. The runner
  reloads only these new files and requires exact class and probability parity
  across the whole holdout before completing. Load only artifacts you trust.
- `run_metadata.json` records fixed configuration, package versions, split counts,
  roundtrip outcomes, limitations, SHA-256 hashes of experiment sources, and hashes
  of generated artifacts including synthetic data and predictions. JSON uses
  Pydantic serialization. Paths are repository-relative, with no local usernames.
  Its own hash is excluded to avoid a circular reference.
- `comparison.png` shows metric means and every individual seed; dots are not
  confidence intervals. Seeds use newly generated datasets, not independent tests
  of one fitted production model.

The tests protect deterministic generation and holdout IDs, training-only scaler
statistics even when a request exceeds the training range, detection of the
injected skew, artifact hashes and LF output, deterministic CSV replay, and exact
prediction agreement after persistence (including replay from the saved CSV).
There is no legacy application test suite or `scripts/commit-gate` in this clone.
A passing companion suite does not establish API or production correctness.

The three seeds are a limited sensitivity check. No calibration, subgroup,
time-shift, customer-group, deployment, or monitoring study is included. A full
repair would need the raw input schema and feature selection fitted inside the
training boundary, a single persisted transform/model contract, migration of
existing artifacts, and request-path parity tests.

## Primary references

- [scikit-learn: inconsistent preprocessing and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [scikit-learn MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
