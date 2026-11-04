"""Phase 2 / M2 — build modeling matrices, splits, and impact tiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split


def load_features_config(root: Path) -> dict[str, Any]:
    path = root / "configs" / "phase2" / "features.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bool_to_int(series: pd.Series) -> pd.Series:
    def map_one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str):
            low = v.strip().lower()
            if low in {"true", "1", "yes"}:
                return 1
            if low in {"false", "0", "no"}:
                return 0
            return np.nan
        return int(bool(v))

    return series.map(map_one).astype(float)


def prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["id"] = out["id"].astype(str)
    for col in ("is_open_access", "has_fulltext", "recent_paper", "has_doi"):
        if col in out.columns:
            out[col] = _bool_to_int(out[col])
    # Missing fulltext → 0 for modeling (conservative)
    if "has_fulltext" in out.columns:
        out["has_fulltext"] = out["has_fulltext"].fillna(0)
    return out


def assign_splits(
    df: pd.DataFrame,
    stratify_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int,
) -> pd.Series:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    idx = df.index.to_numpy()
    y = df[stratify_col]

    idx_train, idx_temp, _, y_temp = train_test_split(
        idx,
        y,
        test_size=(1.0 - train_frac),
        random_state=random_state,
        stratify=y,
    )
    relative_test = test_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=relative_test,
        random_state=random_state,
        stratify=y_temp,
    )

    split = pd.Series(index=df.index, dtype="object")
    split.loc[idx_train] = "train"
    split.loc[idx_val] = "val"
    split.loc[idx_test] = "test"
    return split


def assign_impact_tier(
    citation_per_year: pd.Series,
    split: pd.Series,
    quantiles: list[float],
) -> tuple[pd.Series, dict[str, float]]:
    """Fit tertile thresholds on TRAIN only; apply to all rows."""
    train_vals = citation_per_year[split == "train"]
    q_low, q_high = train_vals.quantile(quantiles).tolist()

    def bucket(v: float) -> str:
        if v <= q_low:
            return "low"
        if v <= q_high:
            return "medium"
        return "high"

    tiers = citation_per_year.map(bucket)
    thresholds = {"q_low": float(q_low), "q_high": float(q_high)}
    return tiers, thresholds


def build_text_corpus(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [df[c].fillna("").astype(str) for c in columns]
    corpus = parts[0]
    for p in parts[1:]:
        corpus = corpus + " " + p
    return corpus


def fit_tfidf(
    train_corpus: pd.Series,
    cfg_text: dict[str, Any],
) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        max_features=int(cfg_text["max_features"]),
        ngram_range=tuple(cfg_text["ngram_range"]),
        min_df=int(cfg_text["min_df"]),
        stop_words=cfg_text.get("stop_words") or None,
    )
    vectorizer.fit(train_corpus.tolist())
    return vectorizer


def tfidf_frame(vectorizer: TfidfVectorizer, corpus: pd.Series, index: pd.Index) -> pd.DataFrame:
    matrix = vectorizer.transform(corpus.tolist())
    names = [f"tfidf_{n}" for n in vectorizer.get_feature_names_out()]
    return pd.DataFrame(matrix.toarray(), columns=names, index=index)


def one_hot_oa_category(series: pd.Series) -> pd.DataFrame:
    cats = ["fully_open", "partially_open", "closed"]
    data = {f"oa_cat_{c}": (series == c).astype(int) for c in cats}
    return pd.DataFrame(data, index=series.index)


def build_track_matrix(
    base: pd.DataFrame,
    structural: list[str],
    tfidf_df: pd.DataFrame | None,
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = [base[structural].astype(float)]
    if extra is not None and len(extra.columns):
        frames.append(extra)
    if tfidf_df is not None:
        frames.append(tfidf_df)
    return pd.concat(frames, axis=1)


def select_features_mi(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    random_state: int,
) -> tuple[list[str], pd.DataFrame]:
    k_use = min(k, X_train.shape[1])
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(
            X, y, random_state=random_state
        ),
        k=k_use,
    )
    selector.fit(X_train.fillna(0.0), y_train)
    scores = pd.DataFrame(
        {
            "feature": X_train.columns,
            "mutual_info": selector.scores_,
            "selected": selector.get_support(),
        }
    ).sort_values("mutual_info", ascending=False)
    selected = X_train.columns[selector.get_support()].tolist()
    return selected, scores


def export_split_csvs(
    base: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Write train/val/test CSVs with ids, labels, split, and both track matrices."""
    for split_name in ("train", "val", "test"):
        mask = base["split"] == split_name
        chunk = base.loc[
            mask,
            [
                "id",
                "split",
                "oa_category",
                "impact_tier",
                "publication_year",
                "paper_age",
                "title_length",
                "abstract_length",
                "keyword_count",
                "concept_count",
                "recent_paper",
                "has_doi",
                "is_open_access",
                "has_fulltext",
                "cited_by_count",
                "citation_per_year",
                "citation_log",
            ],
        ].copy()
        for track_name, mat in matrices.items():
            prefixed = mat.loc[mask].add_prefix(f"{track_name}__")
            chunk = pd.concat([chunk.reset_index(drop=True), prefixed.reset_index(drop=True)], axis=1)
        chunk.to_csv(out_dir / f"{split_name}.csv", index=False)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def build_m2_artifacts(root: Path) -> dict[str, Any]:
    cfg = load_features_config(root)
    ml_dir = root / cfg["paths"]["ml_dir"]
    ml_dir.mkdir(parents=True, exist_ok=True)
    (root / cfg["paths"]["metadata_dir"]).mkdir(parents=True, exist_ok=True)

    source = root / cfg["paths"]["source_csv"]
    raw = pd.read_csv(source)
    base = prepare_base_frame(raw)

    split_cfg = cfg["splits"]
    base["split"] = assign_splits(
        base,
        stratify_col=split_cfg["stratify_column"],
        train_frac=float(split_cfg["train"]),
        val_frac=float(split_cfg["val"]),
        test_frac=float(split_cfg["test"]),
        random_state=int(split_cfg["random_state"]),
    )

    tiers, thresholds = assign_impact_tier(
        base["citation_per_year"],
        base["split"],
        quantiles=list(cfg["impact_tier"]["quantiles"]),
    )
    base["impact_tier"] = tiers

    tfidf_df = None
    vectorizer = None
    text_cfg = cfg["text_features"]
    if text_cfg.get("enabled", True):
        corpus = build_text_corpus(base, text_cfg["columns"])
        train_corpus = corpus[base["split"] == "train"]
        vectorizer = fit_tfidf(train_corpus, text_cfg)
        tfidf_df = tfidf_frame(vectorizer, corpus, base.index)
        joblib.dump(vectorizer, root / cfg["paths"]["vectorizer"])

    track_oa = cfg["tracks"]["oa_category"]
    track_imp = cfg["tracks"]["impact_tier"]

    X_oa = build_track_matrix(
        base,
        track_oa["structural_features"],
        tfidf_df,
        extra=None,
    )
    oa_onehot = one_hot_oa_category(base["oa_category"])
    X_imp = build_track_matrix(
        base,
        track_imp["structural_features"],
        tfidf_df,
        extra=oa_onehot,
    )

    # Feature selection on primary track (train only)
    selection_cfg = cfg["feature_selection"]
    mi_scores = None
    selected_oa = list(X_oa.columns)
    if selection_cfg.get("enabled", True):
        train_mask = base["split"] == "train"
        selected_oa, mi_scores = select_features_mi(
            X_oa.loc[train_mask],
            base.loc[train_mask, "oa_category"],
            k=int(selection_cfg["k_best"]),
            random_state=int(selection_cfg["random_state"]),
        )
        mi_scores.to_csv(ml_dir / "feature_selection_scores_oa.csv", index=False)

    X_oa_selected = X_oa[selected_oa]

    # Persist matrices
    feature_matrix = base[
        [
            "id",
            "split",
            "oa_category",
            "impact_tier",
            "publication_year",
            "paper_age",
            "title_length",
            "abstract_length",
            "keyword_count",
            "concept_count",
            "recent_paper",
            "has_doi",
            "is_open_access",
            "has_fulltext",
            "cited_by_count",
            "citation_per_year",
            "citation_log",
        ]
    ].copy()
    feature_matrix.to_parquet(ml_dir / "feature_matrix.parquet", index=False)
    feature_matrix.to_csv(ml_dir / "feature_matrix.csv", index=False)

    X_oa.to_parquet(ml_dir / "X_oa_category_full.parquet", index=False)
    X_oa_selected.to_parquet(ml_dir / "X_oa_category_selected.parquet", index=False)
    X_imp.to_parquet(ml_dir / "X_impact_tier_full.parquet", index=False)

    # Align ids for modeling convenience
    for name, mat in [
        ("X_oa_category_full", X_oa),
        ("X_oa_category_selected", X_oa_selected),
        ("X_impact_tier_full", X_imp),
    ]:
        export = mat.copy()
        export.insert(0, "id", base["id"].values)
        export.insert(1, "split", base["split"].values)
        export.to_csv(ml_dir / f"{name}.csv", index=False)

    y_oa = base[["id", "split", "oa_category"]].copy()
    y_imp = base[["id", "split", "impact_tier"]].copy()
    y_oa.to_csv(ml_dir / "y_oa_category.csv", index=False)
    y_imp.to_csv(ml_dir / "y_impact_tier.csv", index=False)

    export_split_csvs(
        base,
        {
            "oa": X_oa_selected,
            "impact": X_imp,
        },
        ml_dir,
    )

    label_maps = {
        "oa_category": {
            "classes": ["fully_open", "partially_open", "closed"],
            "counts_all": base["oa_category"].value_counts().to_dict(),
            "counts_by_split": {
                s: base.loc[base["split"] == s, "oa_category"].value_counts().to_dict()
                for s in ("train", "val", "test")
            },
        },
        "impact_tier": {
            "classes": ["low", "medium", "high"],
            "thresholds_train": thresholds,
            "source_column": cfg["impact_tier"]["source_column"],
            "counts_all": base["impact_tier"].value_counts().to_dict(),
            "counts_by_split": {
                s: base.loc[base["split"] == s, "impact_tier"].value_counts().to_dict()
                for s in ("train", "val", "test")
            },
        },
    }
    write_json(ml_dir / "label_maps.json", label_maps)

    feature_lists = {
        "feature_version": cfg["feature_version"],
        "oa_category": {
            "structural": track_oa["structural_features"],
            "leakage_exclude": track_oa["leakage_exclude"],
            "tfidf_enabled": bool(text_cfg.get("enabled", True)),
            "n_features_full": int(X_oa.shape[1]),
            "n_features_selected": int(X_oa_selected.shape[1]),
            "selected_features": selected_oa,
        },
        "impact_tier": {
            "structural": track_imp["structural_features"]
            + ["oa_cat_fully_open", "oa_cat_partially_open", "oa_cat_closed"],
            "leakage_exclude": track_imp["leakage_exclude"],
            "tfidf_enabled": bool(text_cfg.get("enabled", True)),
            "n_features_full": int(X_imp.shape[1]),
            "all_features": list(X_imp.columns),
        },
        "splits": {
            "fractions": {
                "train": split_cfg["train"],
                "val": split_cfg["val"],
                "test": split_cfg["test"],
            },
            "random_state": split_cfg["random_state"],
            "stratify_column": split_cfg["stratify_column"],
            "counts": base["split"].value_counts().to_dict(),
        },
        "impact_thresholds_train": thresholds,
    }
    write_json(ml_dir / "feature_lists.json", feature_lists)

    summary = {
        "n_rows": int(len(base)),
        "split_counts": base["split"].value_counts().to_dict(),
        "oa_counts": base["oa_category"].value_counts().to_dict(),
        "impact_counts": base["impact_tier"].value_counts().to_dict(),
        "impact_thresholds": thresholds,
        "n_features_oa_full": int(X_oa.shape[1]),
        "n_features_oa_selected": int(X_oa_selected.shape[1]),
        "n_features_impact": int(X_imp.shape[1]),
        "selected_oa_preview": selected_oa[:15],
        "ml_dir": str(ml_dir).replace("\\", "/"),
    }
    write_json(ml_dir / "m2_build_summary.json", summary)

    return {
        "base": base,
        "X_oa": X_oa,
        "X_oa_selected": X_oa_selected,
        "X_imp": X_imp,
        "label_maps": label_maps,
        "feature_lists": feature_lists,
        "summary": summary,
        "mi_scores": mi_scores,
        "cfg": cfg,
    }
