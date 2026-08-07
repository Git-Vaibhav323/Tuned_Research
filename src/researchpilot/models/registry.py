"""Model factory for ResearchPilot Phase 2 / M4."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier


def build_model(name: str, random_state: int = 42) -> Any:
    """Return an unfitted estimator for the given model key."""
    key = name.lower().strip()

    if key == "dummy":
        return DummyClassifier(strategy="most_frequent", random_state=random_state)

    if key == "logistic_regression":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        )

    if key == "gaussian_nb":
        return GaussianNB()

    if key == "knn":
        return KNeighborsClassifier(n_neighbors=15, weights="distance")

    if key == "linear_svc":
        # Calibrated so predict_proba works for ROC-AUC
        base = LinearSVC(class_weight="balanced", random_state=random_state, max_iter=10000)
        return CalibratedClassifierCV(base, cv=3)

    if key == "svc_rbf":
        return SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )

    if key == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
        )

    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )

    if key == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )

    if key == "adaboost":
        return AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.8,
            random_state=random_state,
        )

    if key == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=3,
            random_state=random_state,
        )

    if key == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
        )

    if key == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )

    if key == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        )

    raise ValueError(f"Unknown model: {name}")


DEFAULT_MODEL_ORDER = [
    "dummy",
    "logistic_regression",
    "gaussian_nb",
    "knn",
    "linear_svc",
    "svc_rbf",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "adaboost",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "mlp",
]
