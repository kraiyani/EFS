"""Regression checks for the input contract, using synthetic data only."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from experiments.training_serving_skew import (
    FEATURE_NAMES,
    ExperimentConfig,
    RunMetadata,
    evaluate_seed,
    fit_models,
    generate_data,
    run_experiment,
    split_data,
)


def test_generation_and_holdout_are_reproducible():
    config = ExperimentConfig(sample_count=400, tree_count=16)
    first = generate_data(config, 11)
    second = generate_data(config, 11)
    assert_frame_equal(first, second)
    assert not first.equals(generate_data(config, 29))
    split = split_data(first, config, 11)
    repeated = split_data(second, config, 11)
    assert_frame_equal(split.test, repeated.test)
    assert len(split.train) == 300
    assert len(split.test) == 100
    assert set(split.train.row_id).isdisjoint(split.test.row_id)
    assert set(split.train.row_id) | set(split.test.row_id) == set(first.row_id)
    assert set(split.train.target) == {0, 1}
    assert set(split.test.target) == {0, 1}


def test_preprocessing_fits_training_rows_only():
    config = ExperimentConfig(sample_count=400, tree_count=16)
    split = split_data(generate_data(config, 11), config, 11)
    features = split.train.loc[:, FEATURE_NAMES]
    models = fit_models(features, split.train.target, config, 11)
    held_out = split.test.loc[:, FEATURE_NAMES].copy()
    held_out.iloc[0] = features.max() + 100_000
    expected_max = features.max().to_numpy()
    np.testing.assert_array_equal(models.serving_scaler.data_max_, expected_max)
    scaler = models.corrected.named_steps["scale"]
    np.testing.assert_array_equal(scaler.data_max_, expected_max)
    assert scaler.n_samples_seen_ == len(features)
    before = scaler.data_max_.copy()
    models.corrected.predict_proba(held_out)
    np.testing.assert_array_equal(scaler.data_max_, before)
    assert (scaler.transform(held_out)[0] > 1).all()


def test_saved_pipeline_predicts_raw_requests_identically(tmp_path: Path):
    config = ExperimentConfig(sample_count=400, tree_count=16)
    split = split_data(generate_data(config, 11), config, 11)
    models = fit_models(
        split.train.loc[:, FEATURE_NAMES], split.train.target, config, 11
    )
    requests = split.test.loc[:, FEATURE_NAMES]
    path = tmp_path / "pipeline.joblib"
    joblib.dump(models.corrected, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(
        restored.predict(requests), models.corrected.predict(requests)
    )
    np.testing.assert_array_equal(
        restored.predict_proba(requests), models.corrected.predict_proba(requests)
    )


def test_scaled_requests_break_raw_model_contract():
    config = ExperimentConfig(sample_count=400, tree_count=16)
    data = generate_data(config, 11)
    split = split_data(data, config, 11)
    models = fit_models(
        split.train.loc[:, FEATURE_NAMES], split.train.target, config, 11
    )
    predictions, metrics = evaluate_seed(split, models, 11)
    assert predictions.row_id.tolist() == split.test.row_id.tolist()
    assert predictions.target.tolist() == split.test.target.tolist()
    np.testing.assert_array_equal(
        predictions.raw_probability,
        models.raw.predict_proba(split.test.loc[:, FEATURE_NAMES])[:, 1],
    )
    skew = next(row for row in metrics if row.scenario == "scaled_serving")
    assert skew.disagreement_count > 0
    assert skew.mean_absolute_probability_difference > 0.05
    assert skew.roc_auc == 0.5
    corrected = next(row for row in metrics if row.scenario == "consistent_pipeline")
    assert corrected.roc_auc > skew.roc_auc


def test_runner_records_replayable_artifacts(tmp_path: Path):
    config = ExperimentConfig(seeds=(11,), sample_count=400, tree_count=16)
    run_experiment(tmp_path, config)
    metadata = RunMetadata.model_validate_json(
        (tmp_path / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata.config == config
    assert metadata.persistence_checks == {11: True}
    assert metadata.package_versions["scikit-learn"]
    assert metadata.source_sha256["experiments/training_serving_skew.py"]
    assert metadata.artifact_sha256["synthetic_data.csv"]
    assert metadata.artifact_sha256["predictions.csv"]
    for name, digest in metadata.artifact_sha256.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    saved_data = pd.read_csv(tmp_path / "synthetic_data.csv", dtype=str).astype(
        {name: float for name in FEATURE_NAMES}
    )
    saved_predictions = pd.read_csv(tmp_path / "predictions.csv", dtype=str)
    saved_pipeline = joblib.load(tmp_path / "pipeline_seed_11.joblib")
    saved_holdout = saved_data[saved_data["split"] == "test"]
    assert saved_holdout.row_id.tolist() == saved_predictions.row_id.tolist()
    np.testing.assert_array_equal(
        saved_pipeline.predict_proba(saved_holdout.loc[:, FEATURE_NAMES])[:, 1],
        saved_predictions.corrected_probability.astype(float),
    )
    assert metadata.split_counts[0].train_count == 300
    assert metadata.split_counts[0].test_count == 100
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert len(metrics) == 3
    assert set(metrics.test_count) == {100}
    assert (tmp_path / "comparison.png").read_bytes().startswith(b"\x89PNG")
    for name in ("synthetic_data.csv", "predictions.csv", "metrics.csv"):
        assert b"\r" not in (tmp_path / name).read_bytes()
    first_predictions = (tmp_path / "predictions.csv").read_bytes()
    first_data = (tmp_path / "synthetic_data.csv").read_bytes()
    run_experiment(tmp_path, config)
    assert (tmp_path / "predictions.csv").read_bytes() == first_predictions
    assert (tmp_path / "synthetic_data.csv").read_bytes() == first_data
