"""Phase 2 / M3 — multi-method feature selection and train-ready preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    VarianceThreshold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_m3_config(root: Path) -> dict[str, Any]:
    path = root / "configs" / "phase2" / "feature_selection.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_xy(x_path: Path, y_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    if "id" not in x.columns or "split" not in x.columns:
        raise ValueError(f"{x_path} must contain id and split columns")
    ids = x["id"]
    splits = x["split"]
    features = x.drop(columns=["id", "split"])
    labels = y[target_col]
    if len(features) != len(labels):
        raise ValueError("X/y length mismatch")
    return features, labels, ids, splits


def drop_low_variance(X: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(X.fillna(0.0))
    kept = X.columns[vt.get_support()].tolist()
    dropped = [c for c in X.columns if c not in kept]
    return X[kept], dropped


def rank_mutual_info(X: pd.DataFrame, y: pd.Series, random_state: int) -> pd.Series:
    scores = mutual_info_classif(X.fillna(0.0), y, random_state=random_state)
    return pd.Series(scores, index=X.columns, name="mutual_info").sort_values(ascending=False)


def rank_anova_f(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores, _ = f_classif(X.fillna(0.0), y)
    scores = np.nan_to_num(scores, nan=0.0)
    return pd.Series(scores, index=X.columns, name="anova_f").sort_values(ascending=False)


def rank_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    random_state: int,
) -> pd.Series:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X.fillna(0.0), y)
    return pd.Series(rf.feature_importances_, index=X.columns, name="random_forest").sort_values(
        ascending=False
    )


def rank_rfe_logreg(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    step: int,
    random_state: int,
) -> pd.Series:
    """Return ranking scores: higher = selected earlier / more important."""
    # Scale inside RFE so LogisticRegression converges on mixed-scale features
    X_scaled = StandardScaler().fit_transform(X.fillna(0.0))
    estimator = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=random_state,
    )
    n_features = X.shape[1]
    k_use = min(k, n_features)
    rfe = RFE(estimator=estimator, n_features_to_select=k_use, step=step)
    rfe.fit(X_scaled, y)
    # ranking_: 1 is best
    inv = (n_features + 1) - rfe.ranking_
    return pd.Series(inv.astype(float), index=X.columns, name="rfe_logreg").sort_values(
        ascending=False
    )


def top_k_names(ranking: pd.Series, k: int) -> list[str]:
    return ranking.head(min(k, len(ranking))).index.tolist()


def build_consensus(
    method_tops: dict[str, list[str]],
    mi_ranking: pd.Series,
    k: int,
    min_votes: int,
) -> tuple[list[str], pd.DataFrame]:
    all_feats = sorted({f for tops in method_tops.values() for f in tops})
    rows = []
    for feat in all_feats:
        votes = sum(1 for tops in method_tops.values() if feat in tops)
        rows.append(
            {
                "feature": feat,
                "votes": votes,
                "mutual_info": float(mi_ranking.get(feat, 0.0)),
                "in_mutual_info": feat in method_tops.get("mutual_info", []),
                "in_anova_f": feat in method_tops.get("anova_f", []),
                "in_random_forest": feat in method_tops.get("random_forest", []),
                "in_rfe_logreg": feat in method_tops.get("rfe_logreg", []),
            }
        )
    vote_df = pd.DataFrame(rows).sort_values(
        ["votes", "mutual_info"], ascending=[False, False]
    )

    selected = vote_df.loc[vote_df["votes"] >= min_votes, "feature"].tolist()
    if len(selected) < k:
        for feat in mi_ranking.index:
            if feat not in selected:
                selected.append(feat)
            if len(selected) >= k:
                break
    else:
        selected = selected[:k]
    return selected, vote_df


def fit_structural_scaler(
    X_train: pd.DataFrame,
    structural_cols: list[str],
) -> tuple[StandardScaler | None, list[str]]:
    cols = [c for c in structural_cols if c in X_train.columns]
    if not cols:
        return None, []
    scaler = StandardScaler()
    scaler.fit(X_train[cols].fillna(0.0))
    return scaler, cols


def apply_structural_scaler(
    X: pd.DataFrame,
    scaler: StandardScaler | None,
    cols: list[str],
) -> pd.DataFrame:
    out = X.copy()
    if scaler is None or not cols:
        return out
    out[cols] = scaler.transform(out[cols].fillna(0.0))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run_m3(root: Path) -> dict[str, Any]:
    cfg = load_m3_config(root)
    m3_dir = root / cfg["paths"]["m3_dir"]
    m3_dir.mkdir(parents=True, exist_ok=True)

    sel_cfg = cfg["selection"]
    k = int(sel_cfg["k_best"])
    rs = int(sel_cfg["random_state"])

    # ----- Primary track: oa_category -----
    X_oa, y_oa, ids_oa, splits_oa = _read_xy(
        root / cfg["paths"]["x_oa_full"],
        root / cfg["paths"]["y_oa"],
        target_col="oa_category",
    )
    train_mask = splits_oa == "train"
    X_oa_var, dropped_oa = drop_low_variance(
        X_oa, float(sel_cfg["variance_threshold"])
    )
    X_train = X_oa_var.loc[train_mask]
    y_train = y_oa.loc[train_mask]

    rankings: dict[str, pd.Series] = {}
    method_tops: dict[str, list[str]] = {}

    if "mutual_info" in sel_cfg["methods"]:
        rankings["mutual_info"] = rank_mutual_info(X_train, y_train, rs)
        method_tops["mutual_info"] = top_k_names(rankings["mutual_info"], k)

    if "anova_f" in sel_cfg["methods"]:
        rankings["anova_f"] = rank_anova_f(X_train, y_train)
        method_tops["anova_f"] = top_k_names(rankings["anova_f"], k)

    if "random_forest" in sel_cfg["methods"]:
        rankings["random_forest"] = rank_random_forest(
            X_train,
            y_train,
            n_estimators=int(sel_cfg["rf_n_estimators"]),
            random_state=rs,
        )
        method_tops["random_forest"] = top_k_names(rankings["random_forest"], k)

    if "rfe_logreg" in sel_cfg["methods"]:
        rankings["rfe_logreg"] = rank_rfe_logreg(
            X_train,
            y_train,
            k=k,
            step=int(sel_cfg["rfe_step"]),
            random_state=rs,
        )
        method_tops["rfe_logreg"] = top_k_names(rankings["rfe_logreg"], k)

    mi_rank = rankings.get("mutual_info")
    if mi_rank is None:
        # fallback ranking from first available method
        mi_rank = next(iter(rankings.values()))

    final_features, vote_df = build_consensus(
        method_tops=method_tops,
        mi_ranking=mi_rank,
        k=k,
        min_votes=int(sel_cfg["consensus_min_votes"]),
    )

    # Scoreboard table
    scoreboard = pd.DataFrame({"feature": X_oa_var.columns})
    for name, series in rankings.items():
        scoreboard[name] = scoreboard["feature"].map(series).fillna(0.0)
    scoreboard = scoreboard.merge(
        vote_df[["feature", "votes"]], on="feature", how="left"
    )
    scoreboard["votes"] = scoreboard["votes"].fillna(0).astype(int)
    scoreboard["selected_final"] = scoreboard["feature"].isin(final_features)
    scoreboard = scoreboard.sort_values(
        ["selected_final", "votes", "mutual_info"]
        if "mutual_info" in scoreboard.columns
        else ["selected_final", "votes"],
        ascending=[False, False, False]
        if "mutual_info" in scoreboard.columns
        else [False, False],
    )
    scoreboard.to_csv(m3_dir / "selection_scoreboard_oa.csv", index=False)
    vote_df.to_csv(m3_dir / "selection_votes_oa.csv", index=False)

    # Scaling (structural only) on full OA matrix using train fit
    scaler_oa, scale_cols_oa = fit_structural_scaler(
        X_oa.loc[train_mask],
        cfg["scale_structural"]["oa_category"],
    )
    X_oa_scaled = apply_structural_scaler(X_oa, scaler_oa, scale_cols_oa)
    X_oa_final = X_oa_scaled[final_features]

    # Label encoders
    le_oa = LabelEncoder()
    le_oa.fit(y_train)
    y_oa_enc = pd.Series(le_oa.transform(y_oa), index=y_oa.index, name="oa_category_encoded")

    # Persist OA matrices
    oa_out = X_oa_final.copy()
    oa_out.insert(0, "id", ids_oa.values)
    oa_out.insert(1, "split", splits_oa.values)
    oa_out.to_csv(m3_dir / "X_oa_final.csv", index=False)
    oa_out.to_parquet(m3_dir / "X_oa_final.parquet", index=False)

    y_oa_out = pd.DataFrame(
        {
            "id": ids_oa.values,
            "split": splits_oa.values,
            "oa_category": y_oa.values,
            "oa_category_encoded": y_oa_enc.values,
        }
    )
    y_oa_out.to_csv(m3_dir / "y_oa_final.csv", index=False)

    for split_name in ("train", "val", "test"):
        mask = splits_oa == split_name
        oa_out.loc[mask].to_csv(m3_dir / f"oa_{split_name}.csv", index=False)
        y_oa_out.loc[mask].to_csv(m3_dir / f"oa_{split_name}_y.csv", index=False)

    # ----- Secondary track: impact (scale + encode; keep full feature set for M4/M7) -----
    X_imp, y_imp, ids_imp, splits_imp = _read_xy(
        root / cfg["paths"]["x_impact_full"],
        root / cfg["paths"]["y_impact"],
        target_col="impact_tier",
    )
    train_imp = splits_imp == "train"
    scaler_imp, scale_cols_imp = fit_structural_scaler(
        X_imp.loc[train_imp],
        cfg["scale_structural"]["impact_tier"],
    )
    X_imp_scaled = apply_structural_scaler(X_imp, scaler_imp, scale_cols_imp)

    le_imp = LabelEncoder()
    le_imp.fit(y_imp.loc[train_imp])
    y_imp_enc = pd.Series(
        le_imp.transform(y_imp), index=y_imp.index, name="impact_tier_encoded"
    )

    imp_out = X_imp_scaled.copy()
    imp_out.insert(0, "id", ids_imp.values)
    imp_out.insert(1, "split", splits_imp.values)
    imp_out.to_csv(m3_dir / "X_impact_scaled.csv", index=False)
    imp_out.to_parquet(m3_dir / "X_impact_scaled.parquet", index=False)

    y_imp_out = pd.DataFrame(
        {
            "id": ids_imp.values,
            "split": splits_imp.values,
            "impact_tier": y_imp.values,
            "impact_tier_encoded": y_imp_enc.values,
        }
    )
    y_imp_out.to_csv(m3_dir / "y_impact_final.csv", index=False)

    # Preprocessor bundle for M4
    bundle = {
        "feature_version": cfg["feature_version"],
        "primary_track": "oa_category",
        "final_features": final_features,
        "scaler_oa": scaler_oa,
        "scale_cols_oa": scale_cols_oa,
        "scaler_imp": scaler_imp,
        "scale_cols_imp": scale_cols_imp,
        "label_encoder_oa": le_oa,
        "label_encoder_impact": le_imp,
        "oa_classes": list(le_oa.classes_),
        "impact_classes": list(le_imp.classes_),
        "method_tops": method_tops,
        "dropped_low_variance_oa": dropped_oa,
        "k_best": k,
        "consensus_min_votes": int(sel_cfg["consensus_min_votes"]),
        "random_state": rs,
    }
    joblib.dump(bundle, m3_dir / "preprocessor_bundle.joblib")

    # Method agreement summary
    agreement_rows = []
    for method, tops in method_tops.items():
        overlap = len(set(tops) & set(final_features))
        agreement_rows.append(
            {
                "method": method,
                "k": len(tops),
                "overlap_with_final": overlap,
                "overlap_pct": round(100.0 * overlap / max(len(final_features), 1), 2),
            }
        )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(m3_dir / "method_agreement_oa.csv", index=False)

    summary = {
        "feature_version": cfg["feature_version"],
        "n_features_oa_input": int(X_oa.shape[1]),
        "n_features_after_variance": int(X_oa_var.shape[1]),
        "dropped_low_variance": dropped_oa,
        "k_best": k,
        "n_final_features": len(final_features),
        "final_features": final_features,
        "method_agreement": agreement_rows,
        "oa_classes": list(le_oa.classes_),
        "impact_classes": list(le_imp.classes_),
        "scale_cols_oa": scale_cols_oa,
        "scale_cols_imp": scale_cols_imp,
        "split_counts": splits_oa.value_counts().to_dict(),
        "m3_dir": str(m3_dir).replace("\\", "/"),
    }
    write_json(m3_dir / "m3_selection_summary.json", summary)
    write_json(
        m3_dir / "final_feature_list_oa.json",
        {
            "track": "oa_category",
            "n_features": len(final_features),
            "features": final_features,
            "methods": list(method_tops.keys()),
            "consensus_min_votes": int(sel_cfg["consensus_min_votes"]),
        },
    )

    # Small comparison plot for reports
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(agreement["method"], agreement["overlap_with_final"], color="#2c7fb8")
        ax.set_ylabel(f"Overlap with final top-{k}")
        ax.set_xlabel("Selection method")
        ax.set_title("Feature selection method agreement (OA track)")
        ax.set_ylim(0, k)
        for i, v in enumerate(agreement["overlap_with_final"]):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=9)
        fig.tight_layout()
        fig_path = root / cfg["paths"]["reports_dir"] / "figures"
        fig_path.mkdir(parents=True, exist_ok=True)
        out_png = fig_path / "m3_feature_selection_agreement.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        summary["agreement_plot"] = str(out_png).replace("\\", "/")
    except Exception as exc:  # noqa: BLE001
        summary["agreement_plot_error"] = str(exc)

    write_json(m3_dir / "m3_selection_summary.json", summary)
    return summary
