"""Reproduce a raw-training/scaled-serving mismatch with synthetic data only.

Run from the repository root: python -m experiments.training_serving_skew
No application imports, customer files, existing artifacts, or network access.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import joblib
import matplotlib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt

FEATURE_NAMES = [
    "account_age_months",
    "balance_units",
    "income_units",
    "late_payment_days",
    "payment_count",
    "utilisation_percent",
]
SCENARIOS = ("raw_baseline", "scaled_serving", "consistent_pipeline")
SOURCE_PATHS = (
    "README.md",
    "docs/training_serving_skew.md",
    "experiments/__init__.py",
    "experiments/requirements-training-serving-skew.txt",
    "experiments/training_serving_skew.py",
    "tests/test_training_serving_skew.py",
)


class ExperimentConfig(BaseModel):
    """Declared once; no hyperparameter or seed search is performed."""

    model_config = ConfigDict(frozen=True)
    seeds: tuple[NonNegativeInt, ...] = Field(default=(11, 29, 47), min_length=1)
    sample_count: PositiveInt = Field(default=4000, ge=200)
    test_fraction: float = Field(default=0.25, gt=0, lt=1)
    tree_count: PositiveInt = 160
    max_depth: PositiveInt = 8
    min_leaf_count: PositiveInt = 4
    worker_count: Literal[1] = 1


class MetricRow(BaseModel):
    seed: NonNegativeInt
    scenario: Literal["raw_baseline", "scaled_serving", "consistent_pipeline"]
    test_count: PositiveInt
    target_positive_count: NonNegativeInt
    predicted_positive_count: NonNegativeInt
    roc_auc: float = Field(ge=0, le=1)
    average_precision: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    disagreement_count: NonNegativeInt
    disagreement_rate: float = Field(ge=0, le=1)
    mean_absolute_probability_difference: float = Field(ge=0, le=1)
    max_absolute_probability_difference: float = Field(ge=0, le=1)


class SplitCounts(BaseModel):
    seed: NonNegativeInt
    train_count: PositiveInt
    train_positive_count: NonNegativeInt
    test_count: PositiveInt
    test_positive_count: NonNegativeInt


class RunMetadata(BaseModel):
    config: ExperimentConfig
    feature_names: list[str]
    package_versions: dict[str, str]
    source_sha256: dict[str, str]
    artifact_sha256: dict[str, str]
    persistence_checks: dict[int, bool]
    split_counts: list[SplitCounts]
    limitations: list[str]


@dataclass(frozen=True)
class DataSplit:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FittedModels:
    raw: RandomForestClassifier
    serving_scaler: MinMaxScaler
    corrected: Pipeline


def generate_data(config: ExperimentConfig, seed: int) -> pd.DataFrame:
    """Independent synthetic rows with intentionally positive, unequal units."""
    random = np.random.default_rng(seed)
    latent = random.uniform(-1, 1, size=(config.sample_count, len(FEATURE_NAMES)))
    centers = np.array([96, 55000, 75000, 46, 36, 50])
    half_ranges = np.array([84, 45000, 55000, 44, 30, 45])
    data = pd.DataFrame(centers + latent * half_ranges, columns=pd.Index(FEATURE_NAMES))
    # Arbitrary synthetic relationship; these are not credit-risk assumptions.
    score = (
        -0.6 * latent[:, 0]
        + 1.4 * latent[:, 1]
        - 1.2 * latent[:, 2]
        + 1.8 * latent[:, 3]
        - 0.4 * latent[:, 4]
        + 1.3 * latent[:, 5]
        - 0.4
    )
    probability = 1 / (1 + np.exp(-score))
    data.insert(0, "row_id", np.arange(config.sample_count))
    data["target"] = (random.random(config.sample_count) < probability).astype(int)
    return data


def split_data(data: pd.DataFrame, config: ExperimentConfig, seed: int) -> DataSplit:
    """Split once before fitting anything, preserving IDs for paired comparison."""
    train, test = train_test_split(
        data,
        test_size=config.test_fraction,
        random_state=seed,
        stratify=data.target,
    )
    return DataSplit(
        train=train.sort_values("row_id").reset_index(drop=True),
        test=test.sort_values("row_id").reset_index(drop=True),
    )


def fit_models(
    features: pd.DataFrame,
    target: pd.Series,
    config: ExperimentConfig,
    seed: int,
) -> FittedModels:
    """Both scalers see training rows only, isolating serving inconsistency."""
    parameters = {
        "max_depth": config.max_depth,
        "min_samples_leaf": config.min_leaf_count,
        "n_estimators": config.tree_count,
        "n_jobs": config.worker_count,
        "random_state": seed,
    }
    raw = RandomForestClassifier(**parameters).fit(features, target)
    serving_scaler = MinMaxScaler().fit(features)
    corrected = Pipeline(
        [("scale", MinMaxScaler()), ("forest", RandomForestClassifier(**parameters))]
    ).fit(features, target)
    return FittedModels(raw, serving_scaler, corrected)


def evaluate_seed(
    split: DataSplit, models: FittedModels, seed: int
) -> tuple[pd.DataFrame, list[MetricRow]]:
    """Evaluate all paths on the exact same ordered holdout rows and labels."""
    requests = split.test.loc[:, FEATURE_NAMES]
    scaled_requests = pd.DataFrame(
        models.serving_scaler.transform(requests), columns=pd.Index(FEATURE_NAMES)
    )
    raw_probability = models.raw.predict_proba(requests)[:, 1]
    raw_prediction = models.raw.predict(requests)
    probabilities = (
        raw_probability,
        models.raw.predict_proba(scaled_requests)[:, 1],
        models.corrected.predict_proba(requests)[:, 1],
    )
    predictions = (
        raw_prediction,
        models.raw.predict(scaled_requests),
        models.corrected.predict(requests),
    )
    rows = split.test.loc[:, ["row_id", "target"]].copy()
    rows.insert(0, "seed", seed)
    metrics = []
    for scenario, prefix, probability, prediction in zip(
        SCENARIOS,
        ("raw", "skew", "corrected"),
        probabilities,
        predictions,
        strict=True,
    ):
        rows[f"{prefix}_probability"] = probability
        rows[f"{prefix}_prediction"] = prediction
        difference = np.abs(probability - raw_probability)
        disagreement_count = int(np.count_nonzero(prediction != raw_prediction))
        metrics.append(
            MetricRow(
                seed=seed,
                scenario=scenario,
                test_count=len(requests),
                target_positive_count=int(split.test.target.sum()),
                predicted_positive_count=int(np.count_nonzero(prediction)),
                roc_auc=float(roc_auc_score(split.test.target, probability)),
                average_precision=float(
                    average_precision_score(split.test.target, probability)
                ),
                accuracy=float(accuracy_score(split.test.target, prediction)),
                disagreement_count=disagreement_count,
                disagreement_rate=disagreement_count / len(requests),
                mean_absolute_probability_difference=float(difference.mean()),
                max_absolute_probability_difference=float(difference.max()),
            )
        )
    return rows, metrics


def write_chart(metrics: pd.DataFrame, destination: Path) -> None:
    """Means over declared seeds; dots show individual runs, not confidence bounds."""
    labels = ["Raw baseline", "Scaled serving", "Consistent pipeline"]
    colors = ["#1c4e80", "#ca6f1e", "#158579"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    score_names = ["roc_auc", "average_precision", "accuracy"]
    score_labels = ["ROC-AUC", "Average precision", "Accuracy"]
    positions = np.arange(3)
    for index, (scenario, color) in enumerate(zip(SCENARIOS, colors, strict=True)):
        subset = metrics[metrics.scenario == scenario]
        offset = (index - 1) * 0.25
        means = [subset[name].mean() for name in score_names]
        axes[0].bar(
            positions + offset, means, width=0.23, color=color, label=labels[index]
        )
        for position, name in zip(positions + offset, score_names, strict=True):
            axes[0].scatter(
                [position] * len(subset), subset[name], color="#17202a", s=13, zorder=3
            )
    axes[0].set_xticks(positions, score_labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Holdout score")
    axes[0].set_title("Same holdout, different input contract")
    for index, (scenario, color) in enumerate(zip(SCENARIOS, colors, strict=True)):
        values = metrics.loc[metrics.scenario == scenario, "disagreement_rate"] * 100
        axes[1].bar(index, values.mean(), color=color, width=0.6)
        axes[1].scatter([index] * len(values), values, color="#17202a", s=18, zorder=3)
        axes[1].text(index, values.max() + 1.2, f"{values.mean():.1f}%", ha="center")
    axes[1].set_xticks(positions, labels, rotation=10)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Predictions differing from raw baseline (%)")
    axes[1].set_title("Paired classification disagreement")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3
    )
    figure.suptitle(
        "Synthetic training-serving skew | means and individual fixed seeds"
    )
    figure.savefig(
        destination, dpi=160, metadata={"Software": "EFS synthetic experiment"}
    )
    plt.close(figure)


def run_experiment(destination: Path, config: ExperimentConfig) -> RunMetadata:
    destination.mkdir(parents=True, exist_ok=True)
    prediction_frames = []
    data_frames = []
    metric_rows = []
    counts = []
    persistence_checks = {}
    for seed in config.seeds:
        data = generate_data(config, seed)
        split = split_data(data, config, seed)
        models = fit_models(
            split.train.loc[:, FEATURE_NAMES], split.train.target, config, seed
        )
        predictions, metrics = evaluate_seed(split, models, seed)
        prediction_frames.append(predictions)
        metric_rows.extend(metrics)
        data.insert(0, "seed", seed)
        data["split"] = np.where(data.row_id.isin(split.train.row_id), "train", "test")
        data_frames.append(data)
        counts.append(
            SplitCounts(
                seed=seed,
                train_count=len(split.train),
                train_positive_count=int(split.train.target.sum()),
                test_count=len(split.test),
                test_positive_count=int(split.test.target.sum()),
            )
        )
        model_path = destination / f"pipeline_seed_{seed}.joblib"
        joblib.dump(models.corrected, model_path)
        # Load only the artifact just generated here, never the legacy artifacts.
        restored = joblib.load(model_path)
        requests = split.test.loc[:, FEATURE_NAMES]
        parity = bool(
            np.array_equal(
                restored.predict(requests), models.corrected.predict(requests)
            )
            and np.array_equal(
                restored.predict_proba(requests),
                models.corrected.predict_proba(requests),
            )
        )
        if not parity:
            raise ValueError(f"Persisted pipeline prediction mismatch for seed {seed}")
        persistence_checks[seed] = parity
    metrics_frame = pd.DataFrame([row.model_dump() for row in metric_rows])
    for name, frame in (
        ("metrics", metrics_frame),
        ("predictions", pd.concat(prediction_frames, ignore_index=True)),
        ("synthetic_data", pd.concat(data_frames, ignore_index=True)),
    ):
        frame.to_csv(
            destination / f"{name}.csv",
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
    write_chart(metrics_frame, destination / "comparison.png")
    root = Path(__file__).resolve().parents[1]
    artifact_names = [
        "comparison.png",
        "metrics.csv",
        "predictions.csv",
        "synthetic_data.csv",
    ]
    artifact_names.extend(f"pipeline_seed_{seed}.joblib" for seed in config.seeds)
    metadata = RunMetadata(
        config=config,
        feature_names=FEATURE_NAMES,
        package_versions={
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "joblib",
                    "matplotlib",
                    "numpy",
                    "pandas",
                    "pydantic",
                    "scikit-learn",
                )
            },
        },
        source_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS
            if (root / name).exists()
        },
        artifact_sha256={
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in sorted(artifact_names)
        },
        persistence_checks=persistence_checks,
        split_counts=counts,
        limitations=[
            "Entirely synthetic independent rows; no EFS customer data or existing models used.",
            "Positive raw feature ranges deliberately make scaled requests fall below raw thresholds.",
            "Three fixed seeds are a sensitivity check, not confidence intervals or a production estimate.",
            "Both scalers use training rows only; global fitting and feature-selection leakage are excluded.",
            "Fixed forest, split, generator and 0.5 decision threshold; no tuning or seed search.",
            "A consistent forest may work without scaling; consistency is the intervention, not scaling itself.",
            "The legacy application remains unchanged; this experiment does not validate its API.",
        ],
    )
    (destination / "run_metadata.json").write_text(
        metadata.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/training_serving_skew")
    )
    arguments = parser.parse_args()
    run_experiment(arguments.output, ExperimentConfig())
    print(f"Synthetic results written to {arguments.output}")


if __name__ == "__main__":
    main()
