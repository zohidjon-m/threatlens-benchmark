from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional import fallback
    LGBMClassifier = None


EXCLUDED_COLUMNS = {
    "label",
    "ts",
    "uid",
    "id.orig_h",
    "id.resp_h",
    "id.resp_p",
    "capture_type",
    "source_file",
    "capture_id",
    "group_id",
}


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feature_columns = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    numeric_features = [
        c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c]) and c != "label"
    ]
    categorical_features = [c for c in feature_columns if c not in numeric_features]
    return feature_columns, numeric_features, categorical_features


def make_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = groups.nunique()
    if unique_groups >= 5 and y.nunique() > 1:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(cv.split(X, y, groups=groups))
        return train_idx, test_idx

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    return np.array(train_idx), np.array(test_idx)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_models(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        scale_numeric=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        scale_numeric=False,
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if LGBMClassifier is not None:
        models["lightgbm"] = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        scale_numeric=False,
                    ),
                ),
                (
                    "classifier",
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    return models


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, predictions)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": cm.tolist(),
    }


def _save_confusion_matrix_plot(
    confusion: list[list[int]],
    model_name: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = np.array(confusion)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "malicious"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_{model_name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _extract_feature_importance(model: Pipeline) -> pd.DataFrame | None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_[0])
    else:
        return None

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)
    return importance_df


def _save_feature_importance_plot(
    importance_df: pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_df = importance_df.head(15).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_df["feature"], top_df["importance"])
    ax.set_title(f"Top Features — {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / f"importance_{model_name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(
    feature_df: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, numeric_features, categorical_features = build_feature_lists(feature_df)
    feature_columns = numeric_features + categorical_features

    X = feature_df[feature_columns].copy()
    y = feature_df["label"].astype(int)
    groups = feature_df["group_id"].astype(str)

    train_idx, test_idx = make_split(X=X, y=y, groups=groups, random_state=random_state)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    split_info = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "unique_groups_train": int(groups.iloc[train_idx].nunique()),
        "unique_groups_test": int(groups.iloc[test_idx].nunique()),
    }
    with (output_dir / "split_info.json").open("w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2)

    metrics: list[dict[str, Any]] = []
    models = build_models(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        model_metrics = evaluate_model(pipeline, X_test, y_test)
        model_metrics["model"] = model_name
        metrics.append(model_metrics)

        _save_confusion_matrix_plot(
            confusion=model_metrics["confusion_matrix"],
            model_name=model_name,
            output_dir=output_dir / "figures",
        )

        importance_df = _extract_feature_importance(pipeline)
        if importance_df is not None:
            importance_df.to_csv(output_dir / f"feature_importance_{model_name}.csv", index=False)
            _save_feature_importance_plot(
                importance_df=importance_df,
                model_name=model_name,
                output_dir=output_dir / "figures",
            )

    metrics = sorted(metrics, key=lambda item: item["roc_auc"], reverse=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics
