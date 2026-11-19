from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm

from pipeline_config import RUN_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(RUN_DIR)
SPECTRAL_INPUT_DIR = BASE_DIR / "spectral_outputs"
TOKEN_OUTPUT_DIR = BASE_DIR / "token_outputs"
FIGURES_DIR = BASE_DIR / "figures" / "step7"
METRICS_DIR = BASE_DIR / "metrics" / "step7"
LOGS_DIR = BASE_DIR / "logs"

CHANNEL_NAMES = ("T1", "FLAIR")
# Must match step6's FEATURE_NAMES order - the per-patch features are now
# computed there (see the DISK note in step6) and only loaded here.
FEATURE_NAMES = ("Energy", "Entropy", "Mean", "Variance", "Skewness", "Kurtosis", "MAD")
METADATA_COLUMNS = ["Subject_ID", "Patch_ID", "Hemisphere", "Atlas_Region", "Coordinates"]

RANDOM_SEED = 42
UMAP_MAX_NEIGHBORS = 15
UMAP_MIN_TOKENS = 5
DPI = 300


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step7_spectral_token_generation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_subjects(spectral_dir: Path) -> list[Path]:
    if not spectral_dir.exists():
        return []
    return sorted(
        p for p in spectral_dir.iterdir()
        if p.is_dir()
        and (p / "spectral_features_T1.npy").exists()
        and (p / "spectral_features_FLAIR.npy").exists()
        and (p / "metadata.csv").exists()
    )


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------
def process_subject(subject_dir: Path, logger: logging.Logger) -> Optional[dict]:
    start_time = time.perf_counter()
    subject_id = subject_dir.name

    t1_feats = np.load(subject_dir / "spectral_features_T1.npy")
    flair_feats = np.load(subject_dir / "spectral_features_FLAIR.npy")
    metadata_df = pd.read_csv(subject_dir / "metadata.csv")

    n_meta = len(metadata_df)
    n_t1 = t1_feats.shape[0]
    n_flair = flair_feats.shape[0]
    counts_match = n_meta == n_t1 == n_flair
    missing_patches = max(n_meta, n_t1, n_flair) - min(n_meta, n_t1, n_flair)

    if not counts_match:
        logger.warning(
            "%s: patch count mismatch (metadata=%d, T1=%d, FLAIR=%d) - "
            "using the smallest common count and treating the rest as missing.",
            subject_id, n_meta, n_t1, n_flair,
        )

    n_patches = min(n_meta, n_t1, n_flair)
    rows: list[dict] = []

    for idx in range(n_patches):
        meta_row = metadata_df.iloc[idx]
        t1_features = dict(zip(FEATURE_NAMES, t1_feats[idx].tolist()))
        flair_features = dict(zip(FEATURE_NAMES, flair_feats[idx].tolist()))

        row: dict = {
            "Subject_ID": meta_row.get("Subject_ID", subject_id),
            "Patch_ID": meta_row.get("Patch_ID", idx),
            "Hemisphere": meta_row.get("Hemisphere", "NA"),
            "Atlas_Region": meta_row.get("Atlas_Region", "NA"),
            "Coordinates": meta_row.get("Center_Coordinate", "NA"),
        }
        for name, value in t1_features.items():
            row[f"T1_{name}"] = value
        for name, value in flair_features.items():
            row[f"FLAIR_{name}"] = value
        rows.append(row)

    elapsed = time.perf_counter() - start_time

    return {
        "subject_id": subject_id,
        "rows": rows,
        "elapsed_seconds": elapsed,
        "validation": {
            "Subject_ID": subject_id,
            "N_Metadata_Rows": n_meta,
            "N_T1_Patches": n_t1,
            "N_FLAIR_Patches": n_flair,
            "Counts_Match": counts_match,
            "Missing_Patches": missing_patches,
            "N_Tokens_Generated": n_patches,
        },
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_pca(X_scaled: np.ndarray, meta_df: pd.DataFrame, figures_dir: Path, logger: logging.Logger) -> None:
    if X_scaled.shape[0] < 2:
        logger.warning("Not enough tokens for PCA visualization, skipping.")
        return

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    hemi_codes, hemi_labels = pd.factorize(meta_df["Hemisphere"])
    axes[0].scatter(coords[:, 0], coords[:, 1], c=hemi_codes, cmap="tab10", s=8, alpha=0.7)
    axes[0].set_title("PCA - colored by hemisphere")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    denom = max(len(hemi_labels) - 1, 1)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=plt.cm.tab10(i / denom))
               for i in range(len(hemi_labels))]
    axes[0].legend(handles, list(hemi_labels), fontsize=7, loc="best")

    region_codes, _ = pd.factorize(meta_df["Atlas_Region"])
    sc1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=region_codes, cmap="tab20", s=8, alpha=0.7)
    axes[1].set_title("PCA - colored by atlas region")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    fig.colorbar(sc1, ax=axes[1], label="atlas region (encoded)")

    fig.tight_layout()
    fig.savefig(figures_dir / "pca_projection.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved pca_projection.png")


def plot_umap(X_scaled: np.ndarray, meta_df: pd.DataFrame, figures_dir: Path, logger: logging.Logger) -> None:
    n = X_scaled.shape[0]
    if n < UMAP_MIN_TOKENS:
        logger.warning("Not enough tokens (%d) for UMAP visualization, skipping.", n)
        return

    n_neighbors = min(UMAP_MAX_NEIGHBORS, n - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=RANDOM_SEED)
    coords = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 6))
    hemi_codes, hemi_labels = pd.factorize(meta_df["Hemisphere"])
    ax.scatter(coords[:, 0], coords[:, 1], c=hemi_codes, cmap="tab10", s=8, alpha=0.7)
    ax.set_title("UMAP projection - colored by hemisphere")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    denom = max(len(hemi_labels) - 1, 1)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=plt.cm.tab10(i / denom))
               for i in range(len(hemi_labels))]
    ax.legend(handles, list(hemi_labels), fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(figures_dir / "umap_projection.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved umap_projection.png")


def plot_correlation_matrix(feature_df: pd.DataFrame, feature_columns: list[str], figures_dir: Path, logger: logging.Logger) -> None:
    corr = feature_df[feature_columns].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feature_columns)))
    ax.set_yticklabels(feature_columns, fontsize=7)
    ax.set_title("Feature correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures_dir / "correlation_matrix.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved correlation_matrix.png")


def plot_feature_histograms(feature_df: pd.DataFrame, feature_columns: list[str], figures_dir: Path, logger: logging.Logger) -> None:
    n_features = len(feature_columns)
    cols = 4
    rows = int(np.ceil(n_features / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = np.atleast_2d(axes)

    for i, feat in enumerate(feature_columns):
        r, c = divmod(i, cols)
        axes[r, c].hist(feature_df[feat].to_numpy(), bins=40, color="steelblue", edgecolor="black")
        axes[r, c].set_title(feat, fontsize=8)
        axes[r, c].tick_params(labelsize=6)

    for j in range(n_features, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(figures_dir / "feature_histograms.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved feature_histograms.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for d in (TOKEN_OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(LOGS_DIR / "step7.log")
    logger.info("Starting Step 7: Spectral Anatomical Token Generation")

    subject_dirs = discover_subjects(SPECTRAL_INPUT_DIR)
    if not subject_dirs:
        logger.error("No subjects with wavelet coefficients found in %s", SPECTRAL_INPUT_DIR)
        return

    n_subjects = len(subject_dirs)
    all_rows: list[dict] = []
    validation_rows: list[dict] = []
    runtime_rows: list[dict] = []
    n_failed = 0

    pipeline_start = time.perf_counter()

    for i, subject_dir in enumerate(tqdm(subject_dirs, desc="Subjects"), start=1):
        logger.info("Processing subject %d / %d (%s)", i, n_subjects, subject_dir.name)
        try:
            result = process_subject(subject_dir, logger)
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.error("Failed subject %s: %s", subject_dir.name, exc)
            n_failed += 1
            continue

        if result is None:
            continue

        all_rows.extend(result["rows"])
        validation_rows.append(result["validation"])
        runtime_rows.append({
            "subject_id": result["subject_id"],
            "elapsed_seconds": round(result["elapsed_seconds"], 3),
        })

    total_runtime = time.perf_counter() - pipeline_start
    runtime_rows.append({"subject_id": "TOTAL", "elapsed_seconds": round(total_runtime, 3)})

    if not all_rows:
        logger.error("No tokens were generated from any subject.")
        pd.DataFrame(validation_rows).to_csv(METRICS_DIR / "validation.csv", index=False)
        pd.DataFrame(runtime_rows).to_csv(METRICS_DIR / "runtime.csv", index=False)
        return

    token_df = pd.DataFrame(all_rows)
    feature_columns = [c for c in token_df.columns if c not in METADATA_COLUMNS]

    # --- Validation: NaN detection ---
    nan_counts = token_df[feature_columns].isna().sum()
    total_nan = int(nan_counts.sum())
    rows_with_nan = int(token_df[feature_columns].isna().any(axis=1).sum())
    if total_nan:
        logger.warning("Found %d NaN feature values across %d token rows.", total_nan, rows_with_nan)

    # --- Validation: duplicate detection ---
    duplicate_keys = int(token_df.duplicated(subset=["Subject_ID", "Patch_ID"]).sum())
    duplicate_feature_rows = int(token_df.duplicated(subset=feature_columns).sum())
    if duplicate_keys:
        logger.warning("Found %d duplicate (Subject_ID, Patch_ID) token rows.", duplicate_keys)
    if duplicate_feature_rows:
        logger.warning("Found %d tokens with fully duplicate feature vectors.", duplicate_feature_rows)

    # --- Save outputs: CSV + NumPy ---
    token_df.to_csv(TOKEN_OUTPUT_DIR / "tokens.csv", index=False)
    np.save(TOKEN_OUTPUT_DIR / "tokens.npy", token_df[feature_columns].to_numpy(dtype=np.float32))
    with (TOKEN_OUTPUT_DIR / "feature_names.json").open("w") as f:
        json.dump(feature_columns, f, indent=2)
    logger.info("Saved %d tokens (%d features each) -> %s", len(token_df), len(feature_columns), TOKEN_OUTPUT_DIR)

    # --- Metrics: summary ---
    pd.DataFrame([{
        "n_subjects_found": n_subjects,
        "n_subjects_processed": len(validation_rows),
        "n_subjects_failed": n_failed,
        "n_tokens_total": len(token_df),
        "n_features": len(feature_columns),
        "n_nan_values": total_nan,
        "n_rows_with_nan": rows_with_nan,
        "n_duplicate_keys": duplicate_keys,
        "n_duplicate_feature_rows": duplicate_feature_rows,
    }]).to_csv(METRICS_DIR / "summary.csv", index=False)

    # --- Metrics: validation (per-subject missing-patch consistency) ---
    pd.DataFrame(validation_rows).to_csv(METRICS_DIR / "validation.csv", index=False)

    # --- Metrics: statistics (per-feature descriptive stats) ---
    statistics_rows = []
    for col in feature_columns:
        series = token_df[col]
        statistics_rows.append({
            "Feature": col,
            "Count": int(series.notna().sum()),
            "Mean": float(series.mean()) if series.notna().any() else 0.0,
            "Std": float(series.std()) if series.notna().any() else 0.0,
            "Min": float(series.min()) if series.notna().any() else 0.0,
            "Max": float(series.max()) if series.notna().any() else 0.0,
            "Median": float(series.median()) if series.notna().any() else 0.0,
        })
    pd.DataFrame(statistics_rows).to_csv(METRICS_DIR / "statistics.csv", index=False)

    # --- Metrics: runtime ---
    pd.DataFrame(runtime_rows).to_csv(METRICS_DIR / "runtime.csv", index=False)

    # --- Visualizations (NaN rows excluded, PCA/UMAP/corr can't handle them) ---
    clean_df = token_df.dropna(subset=feature_columns)
    n_dropped = len(token_df) - len(clean_df)
    if n_dropped:
        logger.warning("Excluding %d NaN-containing tokens from visualizations.", n_dropped)

    if len(clean_df) >= 2:
        X_scaled = StandardScaler().fit_transform(clean_df[feature_columns].to_numpy(dtype=np.float64))
        plot_pca(X_scaled, clean_df, FIGURES_DIR, logger)
        plot_umap(X_scaled, clean_df, FIGURES_DIR, logger)
        plot_correlation_matrix(clean_df, feature_columns, FIGURES_DIR, logger)
        plot_feature_histograms(clean_df, feature_columns, FIGURES_DIR, logger)
    else:
        logger.warning("Not enough clean tokens for visualizations.")

    logger.info(
        "Step 7 complete. Subjects processed: %d/%d (failed: %d) | Tokens generated: %d | Total runtime: %.2f s",
        len(validation_rows), n_subjects, n_failed, len(token_df), total_runtime,
    )


if __name__ == "__main__":
    main()
