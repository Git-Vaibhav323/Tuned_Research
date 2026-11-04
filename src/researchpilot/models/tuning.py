"""Hyperparameter search spaces and M5 tuning runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from researchpilot.models.train_eval import evaluate_split, load_experiment_config, load_xy


def load_tuning_config(root: Path) -> dict[str, Any]:
    path = root / "configs" / "phase2" / "tuning.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def search_space(name: str, random_state: int) -> tuple[Any, dict[str, list]]:
    """Return (base_estimator, param_distributions)."""
    key = name.lower().strip()

    if key == "adaboost":
        est = AdaBoostClassifier(random_state=random_state)
        params = {
            "n_estimators": [50, 100, 150, 200, 300, 400],
            "learning_rate": [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2],
        }
        return est, params

    if key == "extra_trees":
        est = ExtraTreesClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        params = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 8, 12, 16, 24],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
        return est, params

    if key == "gradient_boosting":
        est = GradientBoostingClassifier(random_state=random_state)
        params = {
            "n_estimators": [100, 150, 200, 300],
            "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.7, 0.85, 1.0],
            "min_samples_leaf": [1, 2, 4],
        }
        return est, params

    if key == "logistic_regression":
        est = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
        params = {
            "C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "solver": ["lbfgs", "saga"],
            "penalty": ["l2"],
        }
        return est, params

    if key == "random_forest":
        est = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
        params = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 8, 12, 16, 24],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
        return est, params

    raise ValueError(f"No search space defined for model: {name}")


def _predict_bundle(model, X, y_true, classes: list[str]):
    y_pred = np.array([str(v) for v in model.predict(X)])
    proba = None
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        model_classes = [str(c) for c in model.classes_]
        proba = np.zeros((len(X), len(classes)), dtype=float)
        for i, c in enumerate(model_classes):
            if c in classes:
                proba[:, classes.index(c)] = raw[:, i]
    return evaluate_split(y_true, y_pred, proba, classes), y_pred, proba


def tune_one(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    classes: list[str],
    random_state: int,
    cv_folds: int,
    n_iter: int,
    scoring: str,
) -> dict[str, Any]:
    est, params = search_space(name, random_state)
    search = RandomizedSearchCV(
        estimator=est,
        param_distributions=params,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_folds,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    val_metrics, y_val_pred, y_val_proba = _predict_bundle(best, X_val, y_val, classes)

    return {
        "model_name": name,
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "cv_results_top": _top_cv_rows(search, k=5),
        "val": val_metrics,
        "best_estimator": best,
        "y_val_pred": y_val_pred,
        "y_val_proba": y_val_proba,
        "search": search,
    }


def _top_cv_rows(search: RandomizedSearchCV, k: int = 5) -> list[dict]:
    df = pd.DataFrame(search.cv_results_)
    df = df.sort_values("rank_test_score").head(k)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "rank": int(r["rank_test_score"]),
                "mean_test_score": float(r["mean_test_score"]),
                "std_test_score": float(r["std_test_score"]),
                "params": r["params"],
            }
        )
    return rows


def run_m5(root: Path) -> dict[str, Any]:
    cfg = load_tuning_config(root)
    exp = load_experiment_config(root)
    paths = exp["paths"]

    out_dir = root / cfg["paths"]["out_dir"]
    ckpt_dir = root / cfg["paths"]["checkpoints"]
    fig_dir = root / cfg["paths"]["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _, feature_cols = load_xy(
        root / paths["x_train"], root / paths["y_train"]
    )
    X_val, y_val, _, _ = load_xy(root / paths["x_val"], root / paths["y_val"])
    X_test, y_test, ids_test, _ = load_xy(root / paths["x_test"], root / paths["y_test"])
    X_val = X_val.reindex(columns=feature_cols, fill_value=0.0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0.0)

    classes = list(exp["primary_task"]["classes"])
    rs = int(cfg["random_state"])
    cv_folds = int(cfg["cv_folds"])
    n_iter = int(cfg["n_iter"])
    scoring = cfg.get("scoring", "f1_macro")
    models = list(cfg["models_to_tune"])

    # Baseline M4 champion metrics (for comparison)
    m4_path = root / cfg["paths"]["m4_summary"]
    m4_baseline = {}
    if m4_path.exists():
        m4_baseline = json.loads(m4_path.read_text(encoding="utf-8"))

    tuned_results = []
    rows = []

    print(f"Tuning {len(models)} models | n_iter={n_iter} | cv={cv_folds} | scoring={scoring}")

    for name in models:
        print(f"  -> {name} ...", flush=True)
        try:
            res = tune_one(
                name=name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                classes=classes,
                random_state=rs,
                cv_folds=cv_folds,
                n_iter=n_iter,
                scoring=scoring,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"     FAILED: {exc}")
            rows.append(
                {
                    "model_name": name,
                    "status": "failed",
                    "error": str(exc),
                    "cv_best_score": None,
                    "val_f1_macro": None,
                    "test_f1_macro": None,
                }
            )
            continue

        tuned_results.append(res)
        # Persist params + val metrics early
        detail = {
            "model_name": name,
            "best_params": _jsonable(res["best_params"]),
            "cv_best_score": res["cv_best_score"],
            "cv_results_top": _jsonable(res["cv_results_top"]),
            "val": {
                k: res["val"][k]
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
        (out_dir / f"tuned_{name}.json").write_text(
            json.dumps(detail, indent=2), encoding="utf-8"
        )
        print(
            f"     CV best={res['cv_best_score']:.4f}  "
            f"val F1={res['val']['f1_macro']:.4f}"
        )

    if not tuned_results:
        raise RuntimeError("All M5 tuning runs failed.")

    # Select champion by validation macro-F1 (not test)
    tuned_results.sort(key=lambda r: r["val"]["f1_macro"], reverse=True)
    champion_name = tuned_results[0]["model_name"]
    champion_params = tuned_results[0]["best_params"]

    # Final test evaluation for every tuned model
    # Optionally refit champion on train+val
    X_tv = pd.concat([X_train, X_val], axis=0)
    y_tv = pd.concat([y_train, y_val], axis=0)

    final_rows = []
    for res in tuned_results:
        name = res["model_name"]
        est_for_test = res["best_estimator"]
        if cfg.get("refit_on_train_val", True) and name == champion_name:
            # Rebuild with best params and fit on train+val
            base, _ = search_space(name, rs)
            base.set_params(**res["best_params"])
            base.fit(X_tv, y_tv)
            est_for_test = base
            res["refit_train_val"] = True
        else:
            res["refit_train_val"] = False

        test_metrics, y_test_pred, y_test_proba = _predict_bundle(
            est_for_test, X_test, y_test, classes
        )
        res["test"] = test_metrics
        res["final_estimator"] = est_for_test

        ckpt = ckpt_dir / f"{name}_tuned.joblib"
        joblib.dump(est_for_test, ckpt)
        res["checkpoint"] = str(ckpt).replace("\\", "/")

        # predictions
        pred_df = pd.DataFrame(
            {
                "id": ids_test.values,
                "y_true": y_test.values,
                "y_pred": y_test_pred,
                "model_name": name,
            }
        )
        if y_test_proba is not None:
            for i, c in enumerate(classes):
                pred_df[f"proba_{c}"] = y_test_proba[:, i]
        pred_df.to_csv(out_dir / f"predictions_test_{name}_tuned.csv", index=False)

        # update detail json with test
        detail_path = out_dir / f"tuned_{name}.json"
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        detail["test"] = {
            k: test_metrics[k]
            for k in (
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "precision_macro",
                "recall_macro",
                "roc_auc_ovr",
                "average_precision_macro",
            )
        }
        detail["refit_train_val"] = res["refit_train_val"]
        detail["confusion_matrix_test"] = test_metrics["confusion_matrix"]
        detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")

        final_rows.append(
            {
                "model_name": name,
                "status": "ok",
                "cv_best_score": res["cv_best_score"],
                "val_accuracy": res["val"]["accuracy"],
                "val_f1_macro": res["val"]["f1_macro"],
                "val_roc_auc_ovr": res["val"]["roc_auc_ovr"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_f1_weighted": test_metrics["f1_weighted"],
                "test_roc_auc_ovr": test_metrics["roc_auc_ovr"],
                "test_average_precision_macro": test_metrics["average_precision_macro"],
                "refit_train_val": res["refit_train_val"],
                "best_params": json.dumps(_jsonable(res["best_params"])),
                "checkpoint": res["checkpoint"],
            }
        )

    leaderboard = pd.DataFrame(final_rows).sort_values(
        "val_f1_macro", ascending=False
    )
    leaderboard.to_csv(out_dir / "tuning_leaderboard.csv", index=False)
    _write_md_leaderboard(out_dir / "tuning_leaderboard.md", leaderboard)

    # Comparison plot: M4 default vs M5 tuned test F1 for overlapping models
    plot_path = _plot_m4_vs_m5(root, leaderboard, fig_dir)

    # DB log
    db_info = _log_tuned_to_db(root, paths["database"], tuned_results, ids_test, y_test, classes)

    champ = next(r for r in tuned_results if r["model_name"] == champion_name)
    summary = {
        "models_tuned": models,
        "n_ok": len(tuned_results),
        "champion_by_val_f1": champion_name,
        "champion_best_params": _jsonable(champion_params),
        "champion_cv_best_score": champ["cv_best_score"],
        "champion_val_f1_macro": champ["val"]["f1_macro"],
        "champion_test_f1_macro": champ["test"]["f1_macro"],
        "champion_test_accuracy": champ["test"]["accuracy"],
        "champion_test_roc_auc_ovr": champ["test"]["roc_auc_ovr"],
        "champion_refit_train_val": champ["refit_train_val"],
        "m4_baseline_champion": m4_baseline.get("champion"),
        "m4_baseline_test_f1_macro": m4_baseline.get("champion_test_f1_macro"),
        "m4_baseline_test_accuracy": m4_baseline.get("champion_test_accuracy"),
        "improvement_test_f1_vs_m4_champion": (
            float(champ["test"]["f1_macro"]) - float(m4_baseline["champion_test_f1_macro"])
            if m4_baseline.get("champion_test_f1_macro") is not None
            else None
        ),
        "leaderboard_csv": str(out_dir / "tuning_leaderboard.csv").replace("\\", "/"),
        "comparison_plot": str(plot_path).replace("\\", "/") if plot_path else None,
        "db_logged_runs": db_info.get("n_runs"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "m5_tuning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _write_md_leaderboard(path: Path, lb: pd.DataFrame) -> None:
    cols = [
        c
        for c in [
            "model_name",
            "cv_best_score",
            "val_f1_macro",
            "test_accuracy",
            "test_f1_macro",
            "test_roc_auc_ovr",
            "refit_train_val",
        ]
        if c in lb.columns
    ]
    lines = [
        "# M5 Tuning Leaderboard — OA category",
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


def _plot_m4_vs_m5(root: Path, m5_lb: pd.DataFrame, fig_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt

        m4_csv = root / "evaluation" / "reports" / "m4" / "leaderboard.csv"
        if not m4_csv.exists():
            return None
        m4 = pd.read_csv(m4_csv)
        m4 = m4[m4["status"] == "ok"][["model_name", "test_f1_macro"]].rename(
            columns={"test_f1_macro": "m4_default_f1"}
        )
        merged = m5_lb[["model_name", "test_f1_macro"]].rename(
            columns={"test_f1_macro": "m5_tuned_f1"}
        ).merge(m4, on="model_name", how="left")
        if merged.empty:
            return None

        x = np.arange(len(merged))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, merged["m4_default_f1"], width, label="M4 default", color="#9ecae1")
        ax.bar(x + width / 2, merged["m5_tuned_f1"], width, label="M5 tuned", color="#2171b5")
        ax.set_xticks(x)
        ax.set_xticklabels(merged["model_name"], rotation=20, ha="right")
        ax.set_ylabel("Test macro-F1")
        ax.set_title("M5 vs M4 — test macro-F1 after hyperparameter tuning")
        ax.legend()
        fig.tight_layout()
        out = fig_dir / "m5_vs_m4_test_f1.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"Plot warning: {exc}")
        return None


def _log_tuned_to_db(
    root: Path,
    db_rel: str,
    results: list[dict],
    ids_test: pd.Series,
    y_test: pd.Series,
    classes: list[str],
) -> dict[str, Any]:
    import uuid

    from researchpilot.db import connect, load_db_config

    db_path = root / db_rel
    if not db_path.exists():
        return {"n_runs": 0, "skipped": True}

    cfg = load_db_config(root)
    conn = connect(db_path, cfg)
    n = 0
    try:
        old = [
            r[0]
            for r in conn.execute(
                """
                SELECT run_id FROM experiment_runs
                WHERE track = 'oa_category' AND notes LIKE 'M5 tuned%'
                """
            ).fetchall()
        ]
        if old:
            conn.executemany("DELETE FROM predictions WHERE run_id = ?", [(i,) for i in old])
            conn.executemany(
                "DELETE FROM experiment_runs WHERE run_id = ?", [(i,) for i in old]
            )
            conn.commit()

        for res in results:
            run_id = f"m5_{res['model_name']}_{uuid.uuid4().hex[:8]}"
            metrics_payload = {
                "best_params": _jsonable(res["best_params"]),
                "cv_best_score": res["cv_best_score"],
                "val_f1_macro": res["val"]["f1_macro"],
                "test": {
                    k: res["test"][k]
                    for k in (
                        "accuracy",
                        "f1_macro",
                        "f1_weighted",
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
                    json.dumps(_jsonable(res["best_params"])),
                    json.dumps(metrics_payload),
                    float(res["cv_best_score"]),
                    "M5 tuned hyperparameters",
                ),
            )
            rows = []
            X_te = pd.read_csv(root / "data" / "ml" / "m3" / "oa_test.csv")
            feat_cols = [c for c in X_te.columns if c not in ("id", "split")]
            X_te = X_te[feat_cols].fillna(0.0)
            y_pred = np.array([str(v) for v in res["final_estimator"].predict(X_te)])
            proba = None
            if hasattr(res["final_estimator"], "predict_proba"):
                raw = res["final_estimator"].predict_proba(X_te)
                model_classes = [str(c) for c in res["final_estimator"].classes_]
                proba = np.zeros((len(X_te), len(classes)), dtype=float)
                for i, c in enumerate(model_classes):
                    if c in classes:
                        proba[:, classes.index(c)] = raw[:, i]

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
                        str(y_pred[i]),
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
