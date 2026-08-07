"""Train / evaluate helpers for Phase 2 / M4."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize

from researchpilot.models.registry import DEFAULT_MODEL_ORDER, build_model


META_COLS = {"id", "split"}


def load_xy(x_path: Path, y_path: Path, target_col: str = "oa_category"):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    ids = X["id"].astype(str)
    feature_cols = [c for c in X.columns if c not in META_COLS]
    X_feat = X[feature_cols].astype(float).fillna(0.0)
    y_lab = y[target_col].astype(str)
    return X_feat, y_lab, ids, feature_cols


def load_m4_config(root: Path) -> dict[str, Any]:
    path = root / "configs" / "phase2" / "models.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(root: Path) -> dict[str, Any]:
    path = root / "configs" / "phase2" / "ml_experiment.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _predict_proba_safe(model, X: pd.DataFrame, classes: list[str]) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Align columns to classes order if needed
        if hasattr(model, "classes_"):
            model_classes = [str(c) for c in model.classes_]
            if list(model_classes) != list(classes):
                aligned = np.zeros((len(X), len(classes)), dtype=float)
                for i, c in enumerate(model_classes):
                    if c in classes:
                        aligned[:, classes.index(c)] = proba[:, i]
                return aligned
        return proba
    return None


def evaluate_split(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    classes: list[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    # sklearn multiclass ROC requires lexicographically sorted labels
    roc_classes = sorted(classes)
    if y_proba is not None:
        try:
            # y_proba columns are expected in `classes` order from caller
            proba_sorted = np.zeros((len(y_true), len(roc_classes)), dtype=float)
            for i, c in enumerate(classes):
                if c in roc_classes:
                    proba_sorted[:, roc_classes.index(c)] = y_proba[:, i]
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(
                    y_true,
                    proba_sorted,
                    multi_class="ovr",
                    average="macro",
                    labels=roc_classes,
                )
            )
            y_bin = label_binarize(y_true, classes=roc_classes)
            metrics["average_precision_macro"] = float(
                average_precision_score(y_bin, proba_sorted, average="macro")
            )
        except Exception:
            metrics["roc_auc_ovr"] = None
            metrics["average_precision_macro"] = None
    else:
        metrics["roc_auc_ovr"] = None
        metrics["average_precision_macro"] = None

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=classes, zero_division=0, output_dict=True
    )
    return metrics


def train_one(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classes: list[str],
    random_state: int,
    cv_folds: int,
    cv_scoring: str,
) -> dict[str, Any]:
    model = build_model(name, random_state=random_state)

    # XGBoost prefers integer labels; map via classes order
    use_int_labels = name.lower() in {"xgboost", "mlp", "lightgbm"}
    if use_int_labels:
        class_to_i = {c: i for i, c in enumerate(classes)}
        y_tr = y_train.map(class_to_i).astype(int)
        y_va = y_val.map(class_to_i).astype(int)
        y_te = y_test.map(class_to_i).astype(int)
        inv = {i: c for c, i in class_to_i.items()}
    else:
        y_tr, y_va, y_te = y_train, y_val, y_test
        inv = None

    cv_scores = cross_val_score(
        build_model(name, random_state=random_state),
        X_train,
        y_tr,
        cv=cv_folds,
        scoring=cv_scoring,
        n_jobs=-1 if name.lower() not in {"svc_rbf", "mlp"} else 1,
    )

    model.fit(X_train, y_tr)

    def _preds(X, y_true_raw):
        raw = model.predict(X)
        if inv is not None:
            y_pred = np.array([inv[int(i)] for i in raw])
        else:
            y_pred = np.array([str(v) for v in raw])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if inv is not None:
                # Trained with integer labels 0..n-1 matching `classes` order
                if proba.shape[1] != len(classes):
                    aligned = np.zeros((len(X), len(classes)), dtype=float)
                    for i, c_idx in enumerate(list(model.classes_)):
                        aligned[:, int(c_idx)] = proba[:, i]
                    proba = aligned
            else:
                proba = _predict_proba_safe(model, X, classes)
        else:
            proba = None
        return evaluate_split(y_true_raw, y_pred, proba, classes), y_pred, proba

    val_metrics, y_val_pred, y_val_proba = _preds(X_val, y_val)
    test_metrics, y_test_pred, y_test_proba = _preds(X_test, y_test)

    return {
        "model_name": name,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "cv_scores": [float(x) for x in cv_scores],
        "val": val_metrics,
        "test": test_metrics,
        "y_val_pred": y_val_pred,
        "y_test_pred": y_test_pred,
        "y_val_proba": y_val_proba,
        "y_test_proba": y_test_proba,
        "estimator": model,
    }


def run_m4(root: Path) -> dict[str, Any]:
    m4_cfg = load_m4_config(root)
    exp_cfg = load_experiment_config(root)
    rs = int(m4_cfg.get("random_state", 42))
    cv_folds = int(m4_cfg.get("cv_folds", 5))
    cv_scoring = m4_cfg.get("cv_scoring", "f1_macro")
    model_names = m4_cfg.get("models") or DEFAULT_MODEL_ORDER

    out_dir = root / m4_cfg["paths"]["out_dir"]
    ckpt_dir = root / m4_cfg["paths"]["checkpoints"]
    fig_dir = root / m4_cfg["paths"]["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = exp_cfg["paths"]
    X_train, y_train, ids_train, feature_cols = load_xy(
        root / paths["x_train"], root / paths["y_train"]
    )
    X_val, y_val, ids_val, _ = load_xy(root / paths["x_val"], root / paths["y_val"])
    X_test, y_test, ids_test, _ = load_xy(root / paths["x_test"], root / paths["y_test"])

    # Align feature columns
    X_val = X_val.reindex(columns=feature_cols, fill_value=0.0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0.0)

    classes = list(exp_cfg["primary_task"]["classes"])
    # Prefer encoder order from y if different alphabetical — use sorted unique from train for stability
    # Keep config order for reporting consistency with Phase 2 design
    present = sorted(y_train.unique().tolist())
    for c in present:
        if c not in classes:
            classes.append(c)

    results = []
    leaderboard_rows = []

    print(f"Training {len(model_names)} models on {X_train.shape[0]} rows x {X_train.shape[1]} features")

    for name in model_names:
        print(f"  -> {name} ...", flush=True)
        try:
            res = train_one(
                name=name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                classes=classes,
                random_state=rs,
                cv_folds=cv_folds,
                cv_scoring=cv_scoring,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"     FAILED: {exc}")
            leaderboard_rows.append(
                {
                    "model_name": name,
                    "status": "failed",
                    "error": str(exc),
                    "cv_f1_macro": None,
                    "val_f1_macro": None,
                    "test_f1_macro": None,
                    "test_accuracy": None,
                    "test_roc_auc_ovr": None,
                }
            )
            continue

        results.append(res)
        leaderboard_rows.append(
            {
                "model_name": name,
                "status": "ok",
                "error": None,
                "cv_f1_macro": res["cv_mean"],
                "cv_f1_macro_std": res["cv_std"],
                "val_accuracy": res["val"]["accuracy"],
                "val_f1_macro": res["val"]["f1_macro"],
                "val_f1_weighted": res["val"]["f1_weighted"],
                "val_roc_auc_ovr": res["val"]["roc_auc_ovr"],
                "test_accuracy": res["test"]["accuracy"],
                "test_precision_macro": res["test"]["precision_macro"],
                "test_recall_macro": res["test"]["recall_macro"],
                "test_f1_macro": res["test"]["f1_macro"],
                "test_f1_weighted": res["test"]["f1_weighted"],
                "test_roc_auc_ovr": res["test"]["roc_auc_ovr"],
                "test_average_precision_macro": res["test"]["average_precision_macro"],
            }
        )

        # Per-model detail JSON (without estimator)
        detail = {
            "model_name": name,
            "cv_mean": res["cv_mean"],
            "cv_std": res["cv_std"],
            "cv_scores": res["cv_scores"],
            "val": {k: v for k, v in res["val"].items()},
            "test": {k: v for k, v in res["test"].items()},
        }
        (out_dir / f"metrics_{name}.json").write_text(
            json.dumps(detail, indent=2), encoding="utf-8"
        )

        # Predictions CSV (test)
        pred_df = pd.DataFrame(
            {
                "id": ids_test.values,
                "y_true": y_test.values,
                "y_pred": res["y_test_pred"],
                "model_name": name,
            }
        )
        if res["y_test_proba"] is not None:
            for i, c in enumerate(classes):
                pred_df[f"proba_{c}"] = res["y_test_proba"][:, i]
        pred_df.to_csv(out_dir / f"predictions_test_{name}.csv", index=False)

    leaderboard = pd.DataFrame(leaderboard_rows)
    if "test_f1_macro" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(
            by=["status", "test_f1_macro", "cv_f1_macro"],
            ascending=[True, False, False],
            na_position="last",
        )
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    leaderboard.to_markdown  # keep import-free; write md manually
    _write_leaderboard_md(out_dir / "leaderboard.md", leaderboard)

    # Save top models
    ok_results = sorted(
        [r for r in results],
        key=lambda r: (r["test"]["f1_macro"], r["cv_mean"]),
        reverse=True,
    )
    top_n = int(m4_cfg.get("save_top_n", 3))
    saved = []
    for res in ok_results[:top_n]:
        path = ckpt_dir / f"{res['model_name']}.joblib"
        joblib.dump(res["estimator"], path)
        saved.append(str(path).replace("\\", "/"))

    # Comparison bar chart
    plot_path = None
    try:
        import matplotlib.pyplot as plt

        ok = leaderboard[leaderboard["status"] == "ok"].copy()
        if len(ok):
            ok = ok.sort_values("test_f1_macro", ascending=True)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(ok["model_name"], ok["test_f1_macro"], color="#2c7fb8")
            ax.set_xlabel("Test macro-F1")
            ax.set_title("M4 — OA category model comparison (test macro-F1)")
            fig.tight_layout()
            plot_path = fig_dir / "m4_model_comparison_f1.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"Plot warning: {exc}")

    # Log to SQLite
    db_summary = _log_to_db(
        root=root,
        db_rel=paths["database"],
        results=ok_results,
        ids_test=ids_test,
        y_test=y_test,
        classes=classes,
    )

    champion = ok_results[0]["model_name"] if ok_results else None
    summary = {
        "n_models_requested": len(model_names),
        "n_models_ok": len(ok_results),
        "n_features": len(feature_cols),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "classes": classes,
        "champion": champion,
        "champion_test_f1_macro": ok_results[0]["test"]["f1_macro"] if ok_results else None,
        "champion_test_accuracy": ok_results[0]["test"]["accuracy"] if ok_results else None,
        "saved_checkpoints": saved,
        "leaderboard_csv": str(out_dir / "leaderboard.csv").replace("\\", "/"),
        "comparison_plot": str(plot_path).replace("\\", "/") if plot_path else None,
        "db_logged_runs": db_summary.get("n_runs"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "m4_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _write_leaderboard_md(path: Path, lb: pd.DataFrame) -> None:
    cols = [
        c
        for c in [
            "model_name",
            "status",
            "cv_f1_macro",
            "val_f1_macro",
            "test_accuracy",
            "test_f1_macro",
            "test_roc_auc_ovr",
        ]
        if c in lb.columns
    ]
    lines = [
        "# M4 Leaderboard — OA category classification",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in lb.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if pd.notna(v) else "")
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_to_db(
    root: Path,
    db_rel: str,
    results: list[dict],
    ids_test: pd.Series,
    y_test: pd.Series,
    classes: list[str],
) -> dict[str, Any]:
    from researchpilot.db import connect, load_db_config

    db_path = root / db_rel
    if not db_path.exists():
        return {"n_runs": 0, "skipped": True, "reason": "db missing"}

    cfg = load_db_config(root)
    conn = connect(db_path, cfg)
    n = 0
    try:
        # Replace prior M4 default runs for a clean leaderboard in DB
        old_ids = [
            r[0]
            for r in conn.execute(
                """
                SELECT run_id FROM experiment_runs
                WHERE track = 'oa_category' AND notes = 'M4 default hyperparameters'
                """
            ).fetchall()
        ]
        if old_ids:
            conn.executemany("DELETE FROM predictions WHERE run_id = ?", [(i,) for i in old_ids])
            conn.executemany(
                "DELETE FROM feature_importance WHERE run_id = ?", [(i,) for i in old_ids]
            )
            conn.executemany(
                "DELETE FROM experiment_runs WHERE run_id = ?", [(i,) for i in old_ids]
            )
            conn.commit()

        for res in results:
            run_id = f"m4_{res['model_name']}_{uuid.uuid4().hex[:8]}"
            metrics_payload = {
                "cv_mean": res["cv_mean"],
                "cv_std": res["cv_std"],
                "val": {
                    k: res["val"][k]
                    for k in (
                        "accuracy",
                        "f1_macro",
                        "f1_weighted",
                        "precision_macro",
                        "recall_macro",
                        "roc_auc_ovr",
                    )
                },
                "test": {
                    k: res["test"][k]
                    for k in (
                        "accuracy",
                        "f1_macro",
                        "f1_weighted",
                        "precision_macro",
                        "recall_macro",
                        "roc_auc_ovr",
                        "average_precision_macro",
                    )
                },
            }
            conn.execute(
                """
                INSERT INTO experiment_runs
                    (run_id, track, model_name, params_json, metrics_json, cv_score, notes)
                VALUES (?, 'oa_category', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    res["model_name"],
                    json.dumps({"source": "m4_default"}),
                    json.dumps(metrics_payload),
                    float(res["cv_mean"]),
                    "M4 default hyperparameters",
                ),
            )
            # Store test predictions
            rows = []
            proba = res["y_test_proba"]
            for i, paper_id in enumerate(ids_test.tolist()):
                proba_json = None
                if proba is not None:
                    proba_json = json.dumps(
                        {classes[j]: float(proba[i, j]) for j in range(len(classes))}
                    )
                rows.append(
                    (
                        run_id,
                        paper_id,
                        str(y_test.iloc[i]),
                        str(res["y_test_pred"][i]),
                        proba_json,
                    )
                )
            conn.executemany(
                """
                INSERT INTO predictions (run_id, paper_id, y_true, y_pred, y_proba_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            n += 1
        conn.commit()
    finally:
        conn.close()
    return {"n_runs": n, "skipped": False}
