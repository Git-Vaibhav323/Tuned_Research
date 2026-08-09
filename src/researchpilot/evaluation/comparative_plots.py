"""Phase 2 / M6 — comparative plots and metric tables."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def load_comparative_config(root: Path) -> dict[str, Any]:
    path = root / "configs" / "phase2" / "comparative.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _proba_cols(classes: list[str]) -> list[str]:
    return [f"proba_{c}" for c in classes]


def load_prediction_csv(path: Path, classes: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    needed = {"y_true", "y_pred"}
    if not needed.issubset(df.columns):
        return None
    pcols = _proba_cols(classes)
    if not all(c in df.columns for c in pcols):
        return None
    return df


def collect_predictions(root: Path, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    classes = list(cfg["classes"])
    out: dict[str, pd.DataFrame] = {}
    m4_dir = root / cfg["paths"]["m4_dir"]
    m5_dir = root / cfg["paths"]["m5_dir"]

    for name in cfg["focus_models"]["m4"]:
        df = load_prediction_csv(m4_dir / f"predictions_test_{name}.csv", classes)
        if df is not None:
            out[f"m4_{name}"] = df

    for name in cfg["focus_models"]["m5"]:
        # files already include _tuned suffix in name
        df = load_prediction_csv(m5_dir / f"predictions_test_{name}.csv", classes)
        if df is not None:
            out[f"m5_{name}"] = df
    return out


def metrics_from_preds(df: pd.DataFrame, classes: list[str]) -> dict[str, Any]:
    y_true = df["y_true"].astype(str)
    y_pred = df["y_pred"].astype(str)
    proba = df[_proba_cols(classes)].to_numpy(dtype=float)
    roc_classes = sorted(classes)
    proba_sorted = np.zeros_like(proba)
    for i, c in enumerate(classes):
        proba_sorted[:, roc_classes.index(c)] = proba[:, i]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr": float(
            roc_auc_score(
                y_true, proba_sorted, multi_class="ovr", average="macro", labels=roc_classes
            )
        ),
        "average_precision_macro": float(
            average_precision_score(
                label_binarize(y_true, classes=roc_classes), proba_sorted, average="macro"
            )
        ),
    }


def plot_confusion_matrices(
    preds: dict[str, pd.DataFrame],
    classes: list[str],
    out_dir: Path,
) -> list[str]:
    paths = []
    n = len(preds)
    if n == 0:
        return paths

    # Individual CMs
    for name, df in preds.items():
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=classes)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"Confusion Matrix — {name}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        p = out_dir / f"cm_{name}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p).replace("\\", "/"))

    # Grid of top models (up to 6)
    items = list(preds.items())[:6]
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (name, df) in zip(axes, items):
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=classes)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            cbar=False,
        )
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle("Comparative Confusion Matrices (test set)", fontsize=12)
    fig.tight_layout()
    grid_path = out_dir / "cm_grid_comparative.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    paths.append(str(grid_path).replace("\\", "/"))
    return paths


def plot_roc_ovr(
    preds: dict[str, pd.DataFrame],
    classes: list[str],
    out_dir: Path,
) -> list[str]:
    """One-vs-rest ROC: one figure per class with all models, plus micro-average style macro curves."""
    paths = []
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(preds), 1)))

    # Per-class OvR ROC across models
    for cls in classes:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for (name, df), color in zip(preds.items(), colors):
            y_true = (df["y_true"].astype(str) == cls).astype(int)
            scores = df[f"proba_{cls}"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC (One-vs-Rest) — class: {cls}")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"roc_ovr_{cls}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p).replace("\\", "/"))

    # Macro ROC-AUC bar comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names, aucs = [], []
    for name, df in preds.items():
        m = metrics_from_preds(df, classes)
        names.append(name)
        aucs.append(m["roc_auc_ovr"])
    order = np.argsort(aucs)
    names = [names[i] for i in order]
    aucs = [aucs[i] for i in order]
    ax.barh(names, aucs, color="#2171b5")
    ax.set_xlabel("Macro ROC-AUC (OvR)")
    ax.set_title("Comparative Macro ROC-AUC (test)")
    ax.set_xlim(0.45, max(0.75, max(aucs) + 0.05) if aucs else 1)
    for i, v in enumerate(aucs):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    p = out_dir / "roc_auc_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p).replace("\\", "/"))
    return paths


def plot_pr_curves(
    preds: dict[str, pd.DataFrame],
    classes: list[str],
    out_dir: Path,
) -> list[str]:
    paths = []
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(preds), 1)))

    for cls in classes:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for (name, df), color in zip(preds.items(), colors):
            y_true = (df["y_true"].astype(str) == cls).astype(int)
            scores = df[f"proba_{cls}"].to_numpy(dtype=float)
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax.plot(recall, precision, color=color, lw=1.8, label=f"{name} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall — class: {cls}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"pr_{cls}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p).replace("\\", "/"))

    # Macro AP comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names, aps = [], []
    for name, df in preds.items():
        m = metrics_from_preds(df, classes)
        names.append(name)
        aps.append(m["average_precision_macro"])
    order = np.argsort(aps)
    names = [names[i] for i in order]
    aps = [aps[i] for i in order]
    ax.barh(names, aps, color="#41ab5d")
    ax.set_xlabel("Macro Average Precision")
    ax.set_title("Comparative Macro Average Precision (test)")
    for i, v in enumerate(aps):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    p = out_dir / "pr_ap_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p).replace("\\", "/"))
    return paths


def plot_metric_bars(metrics_df: pd.DataFrame, out_dir: Path) -> str:
    melt = metrics_df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "f1_macro", "roc_auc_ovr", "average_precision_macro"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=melt, x="model", y="score", hue="metric", ax=ax)
    ax.set_title("Comparative Performance Metrics (test set)")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(title="Metric", fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    p = out_dir / "metrics_grouped_bars.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p).replace("\\", "/")


def extract_feature_importance(
    root: Path,
    cfg: dict[str, Any],
    feature_names: list[str],
    out_dir: Path,
    top_k: int = 15,
) -> tuple[list[str], pd.DataFrame]:
    paths = []
    rows = []
    for item in cfg.get("importance_models", []):
        model_path = root / item["path"]
        name = item["name"]
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        if not hasattr(model, "feature_importances_"):
            continue
        importances = np.asarray(model.feature_importances_, dtype=float)
        if len(importances) != len(feature_names):
            # try common attribute
            continue
        order = np.argsort(importances)[::-1][:top_k]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh(
            [feature_names[i] for i in order][::-1],
            importances[order][::-1],
            color="#6baed6",
        )
        ax.set_title(f"Feature Importance — {name}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        p = out_dir / f"feature_importance_{name}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p).replace("\\", "/"))

        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "model": name,
                    "rank": rank,
                    "feature": feature_names[idx],
                    "importance": float(importances[idx]),
                }
            )

    imp_df = pd.DataFrame(rows)
    return paths, imp_df


def run_m6(root: Path) -> dict[str, Any]:
    cfg = load_comparative_config(root)
    fig_dir = root / cfg["paths"]["figures_dir"]
    out_dir = root / cfg["paths"]["out_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = list(cfg["classes"])
    preds = collect_predictions(root, cfg)
    if not preds:
        raise FileNotFoundError(
            "No prediction CSVs with probabilities found. Run M4/M5 first."
        )

    # Metrics table
    metric_rows = []
    for name, df in preds.items():
        m = metrics_from_preds(df, classes)
        m["model"] = name
        metric_rows.append(m)
    metrics_df = pd.DataFrame(metric_rows).sort_values("f1_macro", ascending=False)
    metrics_df.to_csv(out_dir / "comparative_metrics.csv", index=False)
    _write_metrics_md(out_dir / "comparative_metrics.md", metrics_df)

    cm_paths = plot_confusion_matrices(preds, classes, fig_dir)
    roc_paths = plot_roc_ovr(preds, classes, fig_dir)
    pr_paths = plot_pr_curves(preds, classes, fig_dir)
    bars_path = plot_metric_bars(metrics_df, fig_dir)

    # Feature names from test matrix
    X_test = pd.read_csv(root / cfg["paths"]["x_test"])
    feature_names = [c for c in X_test.columns if c not in ("id", "split")]
    imp_paths, imp_df = extract_feature_importance(root, cfg, feature_names, fig_dir)
    if len(imp_df):
        imp_df.to_csv(out_dir / "feature_importance_top.csv", index=False)

    # Optional: log importances for M5 champion into DB
    db_info = _log_importance_to_db(root, imp_df)

    summary = {
        "n_models_plotted": len(preds),
        "models": list(preds.keys()),
        "classes": classes,
        "metrics_csv": str(out_dir / "comparative_metrics.csv").replace("\\", "/"),
        "figures": {
            "confusion_matrices": cm_paths,
            "roc": roc_paths,
            "pr": pr_paths,
            "metrics_bars": bars_path,
            "feature_importance": imp_paths,
        },
        "best_f1_model": metrics_df.iloc[0]["model"] if len(metrics_df) else None,
        "best_f1_macro": float(metrics_df.iloc[0]["f1_macro"]) if len(metrics_df) else None,
        "best_roc_model": metrics_df.sort_values("roc_auc_ovr", ascending=False).iloc[0]["model"]
        if len(metrics_df)
        else None,
        "db_importance_rows": db_info.get("n_rows", 0),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "m6_comparative_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _write_metrics_md(path: Path, df: pd.DataFrame) -> None:
    cols = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr",
        "average_precision_macro",
    ]
    cols = [c for c in cols if c in df.columns]
    lines = [
        "# M6 Comparative Metrics (test set)",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_importance_to_db(root: Path, imp_df: pd.DataFrame) -> dict[str, Any]:
    if imp_df is None or len(imp_df) == 0:
        return {"n_rows": 0}
    db_path = root / "database" / "researchpilot.db"
    if not db_path.exists():
        return {"n_rows": 0, "skipped": True}

    from researchpilot.db import connect, load_db_config

    # Prefer Extra Trees M5 importances
    subset = imp_df[imp_df["model"] == "extra_trees_m5"]
    if subset.empty:
        subset = imp_df
    cfg = load_db_config(root)
    conn = connect(db_path, cfg)
    n = 0
    try:
        run_id = "m6_feature_importance_extra_trees_m5"
        # Ensure a parent experiment_runs row exists
        exists = conn.execute(
            "SELECT 1 FROM experiment_runs WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone()
        if not exists:
            conn.execute(
                """
                INSERT INTO experiment_runs
                    (run_id, track, model_name, params_json, metrics_json, cv_score, notes)
                VALUES (?, 'oa_category', 'extra_trees', '{}', '{}', NULL, 'M6 feature importance snapshot')
                """,
                (run_id,),
            )
        conn.execute("DELETE FROM feature_importance WHERE run_id = ?", (run_id,))
        rows = [
            (run_id, str(r.feature), float(r.importance))
            for r in subset.itertuples()
        ]
        conn.executemany(
            """
            INSERT INTO feature_importance (run_id, feature_name, importance)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        n = len(rows)
    finally:
        conn.close()
    return {"n_rows": n}
