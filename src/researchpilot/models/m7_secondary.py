"""Phase 2 / M7 — impact_tier classification + light topic clustering."""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score

from researchpilot.models.registry import build_model
from researchpilot.models.train_eval import evaluate_split

# Prefer CPU for tree boosters on Windows (avoids intermittent CUDA OOM).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

META = {"id", "split"}
STRUCTURAL_PREFIXES = (
    "publication_",
    "paper_",
    "title_",
    "abstract_",
    "keyword_",
    "concept_",
    "recent_",
    "is_",
    "has_",
    "oa_",
)


def load_m7_config(root: Path) -> dict[str, Any]:
    with (root / "configs" / "phase2" / "m7_secondary.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_impact_xy(root: Path, cfg: dict[str, Any]):
    X = pd.read_csv(root / cfg["paths"]["x_impact"])
    y = pd.read_csv(root / cfg["paths"]["y_impact"])
    # Align by id
    merged = X.merge(y[["id", "impact_tier"]], on="id", how="inner")
    feature_cols = [c for c in X.columns if c not in META]
    # Safety: drop any citation leakage if present
    leak = [
        c
        for c in feature_cols
        if c.startswith("citation") or c in {"cited_by_count", "impact_tier"}
    ]
    if leak:
        feature_cols = [c for c in feature_cols if c not in leak]
    return merged, feature_cols


def _predict_proba_aligned(model, X: pd.DataFrame, classes: list[str]) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X)
    model_classes = [str(c) for c in model.classes_]
    if list(model_classes) == list(classes):
        return proba
    aligned = np.zeros((len(X), len(classes)), dtype=float)
    for i, c in enumerate(model_classes):
        if c in classes:
            aligned[:, classes.index(c)] = proba[:, i]
    return aligned


def train_impact_models(
    merged: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict], dict[str, Any]]:
    classes = list(cfg["impact_task"]["classes"])
    rs = int(cfg["random_state"])
    cv_folds = int(cfg["cv_folds"])
    scoring = cfg.get("cv_scoring", "f1_macro")

    train = merged[merged["split"] == "train"]
    val = merged[merged["split"] == "val"]
    test = merged[merged["split"] == "test"]

    X_train = train[feature_cols].fillna(0.0)
    X_val = val[feature_cols].fillna(0.0)
    X_test = test[feature_cols].fillna(0.0)
    y_train = train["impact_tier"].astype(str)
    y_val = val["impact_tier"].astype(str)
    y_test = test["impact_tier"].astype(str)

    rows = []
    detailed = []
    for name in cfg["models"]:
        print(f"  -> impact/{name} ...", flush=True)
        try:
            use_int = name.lower() in {"xgboost", "mlp", "lightgbm"}
            if use_int:
                class_to_i = {c: i for i, c in enumerate(classes)}
                inv = {i: c for c, i in class_to_i.items()}
                y_tr = y_train.map(class_to_i).astype(int)
            else:
                inv = None
                y_tr = y_train

            # XGBoost: numpy + serial CV avoids Windows CUDA array_interface failures.
            xgb_safe = name.lower() == "xgboost"
            X_fit = X_train.to_numpy(dtype=np.float32) if xgb_safe else X_train
            y_fit = y_tr.to_numpy() if xgb_safe and hasattr(y_tr, "to_numpy") else y_tr
            cv_jobs = 1 if name.lower() in {"mlp", "xgboost"} else -1

            model = build_model(name, random_state=rs)
            cv = cross_val_score(
                build_model(name, random_state=rs),
                X_fit,
                y_fit,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=cv_jobs,
            )
            model.fit(X_fit, y_fit)

            def _eval(X, y_true):
                X_in = X.to_numpy(dtype=np.float32) if xgb_safe else X
                raw = model.predict(X_in)
                if inv is not None:
                    y_pred = np.array([inv[int(i)] for i in raw])
                    proba = model.predict_proba(X_in) if hasattr(model, "predict_proba") else None
                    if proba is not None and proba.shape[1] == len(classes):
                        # columns follow 0..n-1 matching classes order
                        pass
                    elif proba is not None:
                        aligned = np.zeros((len(X), len(classes)), dtype=float)
                        for i, c_idx in enumerate(list(model.classes_)):
                            aligned[:, int(c_idx)] = proba[:, i]
                        proba = aligned
                else:
                    y_pred = np.array([str(v) for v in raw])
                    proba = _predict_proba_aligned(model, X, classes)
                return evaluate_split(y_true, y_pred, proba, classes), y_pred, proba

            val_m, _, _ = _eval(X_val, y_val)
            test_m, y_test_pred, y_test_proba = _eval(X_test, y_test)

            rows.append(
                {
                    "model_name": name,
                    "status": "ok",
                    "cv_f1_macro": float(np.mean(cv)),
                    "cv_f1_macro_std": float(np.std(cv)),
                    "val_f1_macro": val_m["f1_macro"],
                    "val_accuracy": val_m["accuracy"],
                    "test_accuracy": test_m["accuracy"],
                    "test_f1_macro": test_m["f1_macro"],
                    "test_f1_weighted": test_m["f1_weighted"],
                    "test_roc_auc_ovr": test_m["roc_auc_ovr"],
                    "test_average_precision_macro": test_m["average_precision_macro"],
                }
            )
            detailed.append(
                {
                    "model_name": name,
                    "cv_mean": float(np.mean(cv)),
                    "val": val_m,
                    "test": test_m,
                    "estimator": model,
                    "y_test_pred": y_test_pred,
                    "y_test_proba": y_test_proba,
                    "ids_test": test["id"].astype(str).values,
                    "y_test": y_test.values,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"     FAILED: {exc}")
            rows.append(
                {
                    "model_name": name,
                    "status": "failed",
                    "error": str(exc),
                    "cv_f1_macro": None,
                    "test_f1_macro": None,
                }
            )

    lb = pd.DataFrame(rows)
    if "test_f1_macro" in lb.columns:
        # Rank by validation F1 for selection honesty; report test separately
        if "val_f1_macro" in lb.columns:
            lb = lb.sort_values(
                by=["status", "val_f1_macro", "test_f1_macro"],
                ascending=[True, False, False],
                na_position="last",
            )
        else:
            lb = lb.sort_values("test_f1_macro", ascending=False, na_position="last")

    summary = {
        "n_features": len(feature_cols),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "classes": classes,
        "label_counts": merged["impact_tier"].value_counts().to_dict(),
    }
    return lb, detailed, summary


def run_clustering(
    merged: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    fig_dir: Path,
) -> dict[str, Any]:
    cl_cfg = cfg.get("clustering", {})
    if not cl_cfg.get("enabled", True):
        return {"enabled": False}

    rs = int(cfg["random_state"])
    cols = feature_cols
    if cl_cfg.get("use_tfidf_only", True):
        cols = [c for c in feature_cols if c.startswith("tfidf_")]
        if not cols:
            cols = feature_cols

    train_mask = merged["split"] == "train"
    X_all = merged[cols].fillna(0.0)
    X_train = X_all.loc[train_mask]

    best_k = None
    best_score = -1.0
    scores = []
    for k in range(int(cl_cfg["k_min"]), int(cl_cfg["k_max"]) + 1):
        km = KMeans(n_clusters=k, random_state=rs, n_init=10)
        labels = km.fit_predict(X_train)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(X_train, labels))
        scores.append({"k": k, "silhouette_train": sil})
        if sil > best_score:
            best_score = sil
            best_k = k

    if best_k is None:
        best_k = int(cl_cfg["k_min"])

    km = KMeans(n_clusters=best_k, random_state=rs, n_init=10)
    km.fit(X_train)
    all_labels = km.predict(X_all)
    merged = merged.copy()
    merged["topic_cluster"] = all_labels

    # PCA for visualization (fit on train)
    pca = PCA(n_components=2, random_state=rs)
    xy_train = pca.fit_transform(X_train)
    xy_all = pca.transform(X_all)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(
        xy_all[:, 0],
        xy_all[:, 1],
        c=all_labels,
        cmap="tab10",
        s=12,
        alpha=0.7,
    )
    ax.set_title(f"M7 Topic Clusters (KMeans k={best_k}, PCA-2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    cluster_png = fig_dir / "m7_topic_clusters_pca.png"
    fig.savefig(cluster_png, dpi=150)
    plt.close(fig)

    # Silhouette vs k
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([r["k"] for r in scores], [r["silhouette_train"] for r in scores], "o-")
    ax.axvline(best_k, color="red", ls="--", label=f"best k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette (train)")
    ax.set_title("M7 Cluster model selection")
    ax.legend()
    fig.tight_layout()
    sil_png = fig_dir / "m7_silhouette_vs_k.png"
    fig.savefig(sil_png, dpi=150)
    plt.close(fig)

    agg_kwargs = {
        "n": ("id", "count"),
        "pct_high": ("impact_tier", lambda s: float((s == "high").mean())),
        "pct_medium": ("impact_tier", lambda s: float((s == "medium").mean())),
        "pct_low": ("impact_tier", lambda s: float((s == "low").mean())),
    }
    if "abstract_length" in merged.columns:
        agg_kwargs["mean_abstract_length"] = ("abstract_length", "mean")
    if "concept_count" in merged.columns:
        agg_kwargs["mean_concept_count"] = ("concept_count", "mean")

    profile = merged.groupby("topic_cluster").agg(**agg_kwargs).reset_index()

    return {
        "enabled": True,
        "best_k": best_k,
        "silhouette_train": best_score,
        "silhouette_curve": scores,
        "n_tfidf_features": len(cols),
        "cluster_png": str(cluster_png).replace("\\", "/"),
        "silhouette_png": str(sil_png).replace("\\", "/"),
        "profile": profile,
        "labels": merged[["id", "split", "impact_tier", "topic_cluster"]],
        "model": km,
        "pca": pca,
        "feature_cols_used": cols,
        "merged_with_clusters": merged,
    }


def _feature_group(name: str) -> str:
    if name.startswith("tfidf_"):
        return "tfidf_textual"
    if name.startswith("oa_cat_") or name in {"is_open_access", "has_fulltext"}:
        return "structural_metadata"
    if name.startswith(STRUCTURAL_PREFIXES) or name in {
        "publication_year",
        "paper_age",
        "title_length",
        "abstract_length",
        "keyword_count",
        "concept_count",
        "recent_paper",
    }:
        return "engineered_numerical"
    return "other"


def plot_class_distribution(
    y: pd.Series,
    classes: list[str],
    fig_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    counts = y.value_counts().reindex(classes).fillna(0).astype(int)
    total = int(counts.sum())
    pct = (counts / total * 100).round(2)
    dist = pd.DataFrame(
        {
            "impact_tier": classes,
            "count": [int(counts[c]) for c in classes],
            "percentage": [float(pct[c]) for c in classes],
        }
    )
    dist.to_csv(out_dir / "impact_tier_class_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(dist["impact_tier"], dist["count"], color=["#74c476", "#41ab5d", "#238b45"])
    for i, row in dist.iterrows():
        ax.text(
            i,
            row["count"] + max(total * 0.01, 5),
            f"{int(row['count'])} ({row['percentage']:.1f}%)",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("Count")
    ax.set_title("M7 Impact-tier class distribution")
    ax.set_ylim(0, max(dist["count"]) * 1.15)
    fig.tight_layout()
    png = fig_dir / "m7_impact_class_distribution.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return {
        "table": dist,
        "png": str(png).replace("\\", "/"),
        "balanced": bool(pct.max() - pct.min() < 5.0),
    }


def generate_impact_visualizations(
    champ: dict[str, Any],
    lb: pd.DataFrame,
    feature_cols: list[str],
    classes: list[str],
    fig_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    paths: dict[str, str] = {}
    y_true = np.asarray(champ["y_test"], dtype=str)
    y_pred = np.asarray(champ["y_test_pred"], dtype=str)
    proba = champ.get("y_test_proba")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"M7 Impact confusion matrix — {champ['model_name']}")
    fig.tight_layout()
    p = fig_dir / "m7_impact_confusion_matrix.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["confusion_matrix"] = str(p).replace("\\", "/")

    # Model comparison (val + test F1)
    ok = lb[lb["status"] == "ok"].copy()
    ok = ok.sort_values("val_f1_macro", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(ok))
    ax.barh(y_pos - 0.18, ok["val_f1_macro"], height=0.36, label="Val macro-F1", color="#6baed6")
    ax.barh(y_pos + 0.18, ok["test_f1_macro"], height=0.36, label="Test macro-F1", color="#238b45")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ok["model_name"])
    ax.set_xlabel("Macro-F1")
    ax.set_title("M7 Impact-tier model comparison")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = fig_dir / "m7_impact_model_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["model_comparison"] = str(p).replace("\\", "/")

    # ROC / PR if probabilities available
    if proba is not None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for i, cls in enumerate(classes):
            yt = (y_true == cls).astype(int)
            fpr, tpr, _ = roc_curve(yt, proba[:, i])
            auc = roc_auc_score(yt, proba[:, i])
            ax.plot(fpr, tpr, lw=1.8, label=f"{cls} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"M7 Impact ROC (OvR) — {champ['model_name']}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = fig_dir / "m7_impact_roc_auc.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["roc_auc"] = str(p).replace("\\", "/")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        for i, cls in enumerate(classes):
            yt = (y_true == cls).astype(int)
            precision, recall, _ = precision_recall_curve(yt, proba[:, i])
            ap = average_precision_score(yt, proba[:, i])
            ax.plot(recall, precision, lw=1.8, label=f"{cls} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"M7 Impact Precision-Recall — {champ['model_name']}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = fig_dir / "m7_impact_precision_recall.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["precision_recall"] = str(p).replace("\\", "/")

    # Feature importance for champion
    importance_rows = []
    model = champ["estimator"]
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        if len(imp) == len(feature_cols):
            order = np.argsort(imp)[::-1]
            top = order[:20]
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.barh(
                [feature_cols[i] for i in top][::-1],
                imp[top][::-1],
                color="#6baed6",
            )
            ax.set_xlabel("Importance")
            ax.set_title(f"M7 Feature importance — {champ['model_name']}")
            fig.tight_layout()
            p = fig_dir / "m7_impact_feature_importance.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths["feature_importance"] = str(p).replace("\\", "/")

            for i in order:
                importance_rows.append(
                    {
                        "feature": feature_cols[i],
                        "importance": float(imp[i]),
                        "group": _feature_group(feature_cols[i]),
                    }
                )
            pd.DataFrame(importance_rows).to_csv(
                out_dir / "impact_feature_importance_champion.csv", index=False
            )

    group_share = {}
    if importance_rows:
        gdf = pd.DataFrame(importance_rows)
        group_share = (
            gdf.groupby("group")["importance"].sum().sort_values(ascending=False).to_dict()
        )

    return {
        "paths": paths,
        "top_features": importance_rows[:15],
        "importance_by_group": group_share,
        "confusion_matrix": cm.tolist(),
    }


def _tokenize_field(val: Any) -> list[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    text = str(val).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    for sep in [";", "|", ","]:
        if sep in text:
            return [t.strip().lower() for t in text.split(sep) if t.strip()]
    return [t.strip().lower() for t in text.split() if t.strip()]


def interpret_clusters(
    root: Path,
    clustering: dict[str, Any],
    cfg: dict[str, Any],
    out_dir: Path,
    top_n_terms: int = 8,
    n_titles: int = 3,
) -> list[dict[str, Any]]:
    if not clustering.get("enabled"):
        return []

    labels = clustering["labels"].copy()
    labels["id"] = labels["id"].astype(str)
    source = root / cfg["paths"]["source_csv"]
    papers = pd.read_csv(source, usecols=["id", "title", "keywords_clean", "concepts_clean"])
    papers["id"] = papers["id"].astype(str)
    joined = labels.merge(papers, on="id", how="left")

    # Top TF-IDF tokens from cluster mean vectors
    merged = clustering.get("merged_with_clusters")
    tfidf_cols = clustering.get("feature_cols_used") or []
    centroid_terms: dict[int, list[str]] = {}
    if merged is not None and tfidf_cols:
        for cid, grp in merged.groupby("topic_cluster"):
            means = grp[tfidf_cols].mean().sort_values(ascending=False).head(top_n_terms)
            centroid_terms[int(cid)] = [
                c.replace("tfidf_", "").replace("_", " ") for c in means.index.tolist()
            ]

    total = len(joined)
    summaries = []
    for cid in sorted(joined["topic_cluster"].unique()):
        sub = joined[joined["topic_cluster"] == cid]
        kw_counter: Counter[str] = Counter()
        concept_counter: Counter[str] = Counter()
        for _, row in sub.iterrows():
            kw_counter.update(_tokenize_field(row.get("keywords_clean")))
            concept_counter.update(_tokenize_field(row.get("concepts_clean")))
        titles = (
            sub["title"].dropna().astype(str).head(n_titles).tolist()
            if "title" in sub.columns
            else []
        )
        n = int(len(sub))
        summaries.append(
            {
                "cluster_id": int(cid),
                "n_papers": n,
                "percentage": round(100.0 * n / total, 2) if total else 0.0,
                "dominant_keywords": [w for w, _ in kw_counter.most_common(top_n_terms)],
                "dominant_concepts": [w for w, _ in concept_counter.most_common(top_n_terms)],
                "tfidf_centroid_terms": centroid_terms.get(int(cid), []),
                "representative_titles": titles,
            }
        )

    path = out_dir / "cluster_interpretation.json"
    path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    rows = []
    for s in summaries:
        rows.append(
            {
                "cluster_id": s["cluster_id"],
                "n_papers": s["n_papers"],
                "percentage": s["percentage"],
                "dominant_keywords": "; ".join(s["dominant_keywords"]),
                "dominant_concepts": "; ".join(s["dominant_concepts"]),
                "tfidf_centroid_terms": "; ".join(s["tfidf_centroid_terms"]),
                "representative_titles": " || ".join(s["representative_titles"]),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "cluster_interpretation.csv", index=False)
    return summaries


def run_m7(root: Path) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    cfg = load_m7_config(root)
    out_dir = root / cfg["paths"]["out_dir"]
    ckpt_dir = root / cfg["paths"]["checkpoints"]
    fig_dir = root / cfg["paths"]["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading impact matrices ...")
    merged, feature_cols = _load_impact_xy(root, cfg)
    leak_cols = [
        c
        for c in feature_cols
        if c.startswith("citation") or c in {"cited_by_count", "impact_tier"}
    ]
    print(f"Leakage columns in features: {leak_cols if leak_cols else 'none'}")

    print(f"Training impact_tier models ({len(cfg['models'])}) ...")
    lb, detailed, impact_summary = train_impact_models(merged, feature_cols, cfg)
    lb.to_csv(out_dir / "impact_leaderboard.csv", index=False)
    _write_md(out_dir / "impact_leaderboard.md", lb, "M7 Impact-tier Leaderboard")

    xgb_row = lb[lb["model_name"] == "xgboost"]
    xgb_status = (
        str(xgb_row.iloc[0]["status"])
        if len(xgb_row)
        else "missing"
    )
    xgb_error = None
    if len(xgb_row) and "error" in xgb_row.columns and pd.notna(xgb_row.iloc[0].get("error")):
        xgb_error = str(xgb_row.iloc[0]["error"])

    # Champion by validation F1 (honest selection)
    ok = [d for d in detailed if d.get("val")]
    ok.sort(key=lambda d: d["val"]["f1_macro"], reverse=True)
    champion = ok[0]["model_name"] if ok else None
    champ_row = next((d for d in ok if d["model_name"] == champion), None)

    # Save top checkpoints + predictions
    saved = []
    for res in ok[: int(cfg.get("save_top_n", 3))]:
        path = ckpt_dir / f"impact_{res['model_name']}.joblib"
        joblib.dump(res["estimator"], path)
        saved.append(str(path).replace("\\", "/"))
        pred = pd.DataFrame(
            {
                "id": res["ids_test"],
                "y_true": res["y_test"],
                "y_pred": res["y_test_pred"],
                "model_name": res["model_name"],
            }
        )
        if res["y_test_proba"] is not None:
            for i, c in enumerate(cfg["impact_task"]["classes"]):
                pred[f"proba_{c}"] = res["y_test_proba"][:, i]
        pred.to_csv(out_dir / f"predictions_test_impact_{res['model_name']}.csv", index=False)

        detail = {
            "model_name": res["model_name"],
            "cv_mean": res["cv_mean"],
            "val": {
                k: res["val"][k]
                for k in ("accuracy", "f1_macro", "roc_auc_ovr", "average_precision_macro")
            },
            "test": {
                k: res["test"][k]
                for k in ("accuracy", "f1_macro", "roc_auc_ovr", "average_precision_macro")
            },
        }
        (out_dir / f"metrics_impact_{res['model_name']}.json").write_text(
            json.dumps(detail, indent=2), encoding="utf-8"
        )

    # Class distribution + champion visualizations
    class_dist = plot_class_distribution(
        merged["impact_tier"].astype(str),
        list(cfg["impact_task"]["classes"]),
        fig_dir,
        out_dir,
    )
    viz_info: dict[str, Any] = {}
    if champ_row is not None:
        print("Generating impact-tier visualizations ...")
        viz_info = generate_impact_visualizations(
            champ_row, lb, feature_cols, list(cfg["impact_task"]["classes"]), fig_dir, out_dir
        )

    # Legacy single-metric comparison plot (keep existing filename)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = lb[lb["status"] == "ok"].sort_values("test_f1_macro")
    ax.barh(plot_df["model_name"], plot_df["test_f1_macro"], color="#238b45")
    ax.set_xlabel("Test macro-F1")
    ax.set_title("M7 — Impact-tier model comparison (test macro-F1)")
    fig.tight_layout()
    f1_png = fig_dir / "m7_impact_model_comparison_f1.png"
    fig.savefig(f1_png, dpi=150)
    plt.close(fig)

    print("Running topic clustering ...")
    clustering = run_clustering(merged, feature_cols, cfg, fig_dir)
    cluster_interp: list[dict[str, Any]] = []
    if clustering.get("enabled"):
        clustering["profile"].to_csv(out_dir / "cluster_profiles.csv", index=False)
        clustering["labels"].to_csv(out_dir / "paper_topic_clusters.csv", index=False)
        joblib.dump(
            {
                "kmeans": clustering["model"],
                "pca": clustering["pca"],
                "feature_cols": clustering["feature_cols_used"],
                "best_k": clustering["best_k"],
            },
            ckpt_dir / "topic_kmeans.joblib",
        )
        print("Building cluster interpretation summaries ...")
        cluster_interp = interpret_clusters(root, clustering, cfg, out_dir)

    # DB logging
    db_info = _log_m7_db(root, ok, cfg, clustering)

    n_attempted = int(len(cfg["models"]))
    n_ok = int(len(ok))
    n_failed = n_attempted - n_ok

    summary = {
        "track": "impact_tier",
        "impact": {
            **impact_summary,
            "n_models_attempted": n_attempted,
            "n_models_ok": n_ok,
            "n_models_failed": n_failed,
            "xgboost_status": xgb_status,
            "xgboost_error": xgb_error,
            "champion_by_val_f1": champion,
            "champion_val_f1_macro": champ_row["val"]["f1_macro"] if champ_row else None,
            "champion_test_f1_macro": champ_row["test"]["f1_macro"] if champ_row else None,
            "champion_test_accuracy": champ_row["test"]["accuracy"] if champ_row else None,
            "champion_test_roc_auc_ovr": champ_row["test"]["roc_auc_ovr"] if champ_row else None,
            "saved_checkpoints": saved,
            "leaderboard_csv": str(out_dir / "impact_leaderboard.csv").replace("\\", "/"),
            "comparison_plot": str(f1_png).replace("\\", "/"),
            "class_distribution": class_dist["table"].to_dict(orient="records"),
            "class_distribution_png": class_dist["png"],
            "visualization_paths": viz_info.get("paths", {}),
            "importance_by_group": viz_info.get("importance_by_group", {}),
            "top_features": viz_info.get("top_features", []),
            "leakage_columns_found": leak_cols,
        },
        "clustering": {
            "enabled": clustering.get("enabled", False),
            "best_k": clustering.get("best_k"),
            "silhouette_train": clustering.get("silhouette_train"),
            "n_tfidf_features": clustering.get("n_tfidf_features"),
            "cluster_png": clustering.get("cluster_png"),
            "silhouette_png": clustering.get("silhouette_png"),
            "interpretation": cluster_interp,
        },
        "db_logged_runs": db_info.get("n_runs"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Impact features exclude citation_*; labels from train-only tertiles (M2).",
            "Champion selected by validation macro-F1; test reported once.",
            "Clustering is unsupervised support for Phase 3 topic/gap narrative.",
            "XGBoost forced to CPU (device=cpu, CUDA_VISIBLE_DEVICES=-1, n_jobs=1).",
        ],
    }
    (out_dir / "m7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_md(path: Path, lb: pd.DataFrame, title: str) -> None:
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
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
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


def _log_m7_db(root: Path, results: list[dict], cfg: dict, clustering: dict) -> dict[str, Any]:
    from researchpilot.db import connect, load_db_config

    db_path = root / "database" / "researchpilot.db"
    if not db_path.exists():
        return {"n_runs": 0, "skipped": True}

    conf = load_db_config(root)
    conn = connect(db_path, conf)
    n = 0
    try:
        old = [
            r[0]
            for r in conn.execute(
                "SELECT run_id FROM experiment_runs WHERE track = 'impact_tier' AND notes LIKE 'M7%'"
            ).fetchall()
        ]
        if old:
            conn.executemany("DELETE FROM predictions WHERE run_id = ?", [(i,) for i in old])
            conn.executemany("DELETE FROM experiment_runs WHERE run_id = ?", [(i,) for i in old])
            conn.commit()

        classes = list(cfg["impact_task"]["classes"])
        for res in results:
            run_id = f"m7_impact_{res['model_name']}_{uuid.uuid4().hex[:8]}"
            metrics = {
                "cv_mean": res["cv_mean"],
                "val_f1_macro": res["val"]["f1_macro"],
                "test": {
                    k: res["test"][k]
                    for k in ("accuracy", "f1_macro", "roc_auc_ovr", "average_precision_macro")
                },
            }
            conn.execute(
                """
                INSERT INTO experiment_runs
                    (run_id, track, model_name, params_json, metrics_json, cv_score, notes)
                VALUES (?, 'impact_tier', ?, '{}', ?, ?, 'M7 impact_tier default hyperparameters')
                """,
                (run_id, res["model_name"], json.dumps(metrics), float(res["cv_mean"])),
            )
            rows = []
            proba = res["y_test_proba"]
            for i, paper_id in enumerate(res["ids_test"]):
                proba_json = None
                if proba is not None:
                    proba_json = json.dumps(
                        {classes[j]: float(proba[i, j]) for j in range(len(classes))}
                    )
                rows.append(
                    (
                        run_id,
                        str(paper_id),
                        str(res["y_test"][i]),
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

        # Optional cluster note run
        if clustering.get("enabled"):
            conn.execute(
                """
                INSERT INTO experiment_runs
                    (run_id, track, model_name, params_json, metrics_json, cv_score, notes)
                VALUES (?, 'impact_tier', 'kmeans_topics', ?, ?, ?, 'M7 topic clustering')
                """,
                (
                    f"m7_clusters_{uuid.uuid4().hex[:8]}",
                    json.dumps({"k": clustering["best_k"], "features": "tfidf"}),
                    json.dumps({"silhouette_train": clustering["silhouette_train"]}),
                    float(clustering["silhouette_train"]),
                ),
            )
            n += 1
        conn.commit()
    finally:
        conn.close()
    return {"n_runs": n}
