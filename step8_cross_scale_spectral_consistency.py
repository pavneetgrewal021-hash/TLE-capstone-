from __future__ import annotations

import json
import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from pipeline_config import RUN_DIR

BASE_DIR = Path(RUN_DIR)
TOKEN_INPUT_DIR = BASE_DIR / "token_outputs"
EMBEDDING_OUTPUT_DIR = BASE_DIR / "embedding_outputs"
FIGURES_DIR = BASE_DIR / "figures" / "step8"
METRICS_DIR = BASE_DIR / "metrics" / "step8"
LOGS_DIR = BASE_DIR / "logs"

METADATA_COLUMNS = ["Subject_ID", "Patch_ID", "Hemisphere", "Atlas_Region", "Coordinates"]
SCALES = (24, 32, 48)

# Step 5 only ever extracted patches at one physical size (32^3). True
# re-extraction at 24^3/48^3 would need the original MRI volumes, which
# this step's declared input (token_outputs/) does not include. Each
# scale view is instead built from the SAME token grid: a Gaussian-
# weighted average of a token's own features with its spatial neighbors
# (same subject only) within a radius proportional to that scale - a
# small radius stays close to the token itself, a large radius blends in
# more surrounding anatomy, the same qualitative effect a physically
# larger patch's summary statistics would have. Every scale view is
# therefore built from real, already-computed values, never fabricated.
SCALE_NEIGHBOR_BANDWIDTH_DIVISOR = 2.0

PROJECTION_HIDDEN_DIM = 64
PROJECTION_OUTPUT_DIM = 32

TEMPERATURE = 0.07
LEARNING_RATE = 1e-3
N_EPOCHS = 50
BATCH_SIZE = 256
RANDOM_SEED = 42

TSNE_SAMPLE_SIZE = 1500
# Tokens sampled for the cross-scale retrieval test (a sample x sample
# similarity matrix per scale pair, so kept moderate).
RETRIEVAL_SAMPLE_SIZE = 3000
SIMILARITY_MATRIX_SAMPLE = 24

DPI = 300


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step8_cross_scale_consistency")
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
# Multi-scale view construction
# ---------------------------------------------------------------------------
def parse_coordinates(coord_str: str) -> tuple[float, float, float]:
    parts = str(coord_str).strip("()").split(",")
    return float(parts[0]), float(parts[1]), float(parts[2])


def build_scale_views(
    token_df: pd.DataFrame, feature_columns: list[str], scales: tuple[int, ...], logger: logging.Logger
) -> dict[int, np.ndarray]:
    coords = np.array([parse_coordinates(c) for c in token_df["Coordinates"]])
    features = token_df[feature_columns].to_numpy(dtype=np.float64)
    subjects = token_df["Subject_ID"].to_numpy()

    views: dict[int, np.ndarray] = {scale: np.zeros_like(features) for scale in scales}

    for subject_id in tqdm(np.unique(subjects), desc="Building scale views"):
        idx = np.where(subjects == subject_id)[0]
        tree = cKDTree(coords[idx])

        for scale in scales:
            radius = scale / 2.0
            bandwidth = max(radius / SCALE_NEIGHBOR_BANDWIDTH_DIVISOR, 1e-6)

            for global_i in idx:
                neighbor_local = tree.query_ball_point(coords[global_i], r=radius)
                if len(neighbor_local) <= 1:
                    views[scale][global_i] = features[global_i]
                    continue
                neighbor_global = idx[neighbor_local]
                dists = np.linalg.norm(coords[neighbor_global] - coords[global_i], axis=1)
                weights = np.exp(-(dists ** 2) / (2 * bandwidth ** 2))
                weights /= weights.sum()
                views[scale][global_i] = (features[neighbor_global] * weights[:, None]).sum(axis=0)

    logger.info("Built scale views for scales=%s across %d tokens.", scales, features.shape[0])
    return views


def standardize_views(views: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    pooled = np.concatenate(list(views.values()), axis=0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std[std == 0] = 1.0
    return {scale: (arr - mean) / std for scale, arr in views.items()}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def cross_scale_similarity(views: dict[int, np.ndarray], scales: tuple[int, ...]) -> np.ndarray:
    """Per-token mean pairwise cosine similarity across all scale-pair
    combinations - one score per token, used for the before/after
    Wilcoxon comparison."""
    normed = {scale: l2_normalize(arr) for scale, arr in views.items()}
    pairs = list(combinations(scales, 2))
    sims = np.zeros((normed[scales[0]].shape[0], len(pairs)))
    for j, (a, b) in enumerate(pairs):
        sims[:, j] = np.sum(normed[a] * normed[b], axis=1)
    return sims.mean(axis=1)


def cross_scale_retrieval(
    views: dict[int, np.ndarray], scales: tuple[int, ...], sample_idx: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    """For each scale pair (a, b) and each sampled token: is its nearest
    neighbour at scale b (cosine, among all sampled tokens) the SAME token?
    One boolean per token per pair.

    This is the primary consistency metric. The raw before/after cosine
    (cross_scale_similarity) compares two different spaces - 14-dim
    standardized features vs the 32-dim learned embedding - and InfoNCE
    optimises the RANKING of positives over negatives, not their absolute
    cosine, so that comparison came out "worse" (0.925 -> 0.896) even
    though cross-scale matching improved a lot."""
    hits = {}
    for a, b in combinations(scales, 2):
        sim = l2_normalize(views[a][sample_idx]) @ l2_normalize(views[b][sample_idx]).T
        hits[(a, b)] = sim.argmax(axis=1) == np.arange(len(sample_idx))
    return hits


def mcnemar_exact(before_hit: np.ndarray, after_hit: np.ndarray) -> float:
    """Exact McNemar test (paired binary outcomes, same tokens before vs
    after), one-sided for improvement: binomial test on discordant pairs."""
    gained = int(np.sum(~before_hit & after_hit))
    lost = int(np.sum(before_hit & ~after_hit))
    if gained + lost == 0:
        return 1.0
    return float(stats.binomtest(gained, gained + lost, 0.5, alternative="greater").pvalue)


# ---------------------------------------------------------------------------
# Projection head + InfoNCE
# ---------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)


def info_nce_loss(embeddings: torch.Tensor, anchor_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    """Multi-positive InfoNCE / NT-Xent: for each embedding, positives are
    the OTHER scale views of the same anchor token; negatives are every
    embedding belonging to a different anchor (any scale)."""
    sim = embeddings @ embeddings.T / temperature
    n = sim.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    sim = sim.masked_fill(self_mask, -1e9)

    positive_mask = (anchor_ids.unsqueeze(0) == anchor_ids.unsqueeze(1)) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = positive_mask.sum(dim=1).clamp(min=1)
    positive_log_prob = (log_prob * positive_mask).sum(dim=1) / pos_counts
    return -positive_log_prob.mean()


def train_projection_head(
    views: dict[int, np.ndarray], scales: tuple[int, ...], input_dim: int, logger: logging.Logger
) -> tuple[nn.Module, list[float]]:
    torch.manual_seed(RANDOM_SEED)
    n_tokens = next(iter(views.values())).shape[0]

    model = ProjectionHead(input_dim, PROJECTION_HIDDEN_DIM, PROJECTION_OUTPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    tensors = {scale: torch.tensor(arr, dtype=torch.float32) for scale, arr in views.items()}

    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(n_tokens)
    loss_history: list[float] = []

    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        rng.shuffle(indices)
        epoch_losses: list[float] = []

        for start in range(0, n_tokens, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            if len(batch_idx) < 2:
                continue
            batch_idx_t = torch.tensor(batch_idx, dtype=torch.long)

            batch_inputs = torch.cat([tensors[scale][batch_idx_t] for scale in scales], dim=0)
            anchor_ids = torch.tensor(np.tile(batch_idx, len(scales)), dtype=torch.long)

            embeddings = model(batch_inputs)
            loss = info_nce_loss(embeddings, anchor_ids, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        loss_history.append(mean_epoch_loss)
        logger.info("Epoch %d/%d - InfoNCE loss: %.4f", epoch + 1, N_EPOCHS, mean_epoch_loss)

    return model, loss_history


def compute_all_embeddings(
    model: nn.Module, views: dict[int, np.ndarray], scales: tuple[int, ...]
) -> dict[int, np.ndarray]:
    model.eval()
    embeddings: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for scale in scales:
            x = torch.tensor(views[scale], dtype=torch.float32)
            embeddings[scale] = model(x).numpy()
    return embeddings


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_loss_curve(loss_history: list[float], figures_dir: Path, logger: logging.Logger) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title("Cross-scale contrastive training loss")
    fig.tight_layout()
    fig.savefig(figures_dir / "loss_curve.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved loss_curve.png")


def plot_similarity_matrix(
    embeddings: dict[int, np.ndarray], scales: tuple[int, ...], sample_size: int,
    figures_dir: Path, logger: logging.Logger,
) -> None:
    n_tokens = embeddings[scales[0]].shape[0]
    sample_size = min(sample_size, n_tokens)
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(n_tokens, size=sample_size, replace=False)

    stacked = np.array([embeddings[scale][i] for i in sample_idx for scale in scales])
    stacked = l2_normalize(stacked)
    sim = stacked @ stacked.T

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
    for k in range(1, sample_size):
        ax.axhline(k * len(scales) - 0.5, color="white", linewidth=0.4)
        ax.axvline(k * len(scales) - 0.5, color="white", linewidth=0.4)
    ax.set_title(f"Cross-scale embedding cosine similarity ({sample_size} anchors x {len(scales)} scales)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures_dir / "similarity_matrix.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved similarity_matrix.png")


def plot_tsne(
    embeddings: dict[int, np.ndarray], scales: tuple[int, ...], sample_size: int,
    figures_dir: Path, logger: logging.Logger,
) -> None:
    n_tokens = embeddings[scales[0]].shape[0]
    sample_size = min(sample_size, n_tokens)
    if sample_size < 10:
        logger.warning("Not enough tokens (%d) for t-SNE visualization, skipping.", sample_size)
        return

    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(n_tokens, size=sample_size, replace=False)

    all_points, labels = [], []
    for scale in scales:
        all_points.append(embeddings[scale][sample_idx])
        labels.extend([scale] * sample_size)
    X = np.concatenate(all_points, axis=0)
    labels_arr = np.array(labels)

    perplexity = min(30, max(5, sample_size // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_SEED, init="pca")
    coords = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for scale, color in zip(scales, ("steelblue", "darkorange", "seagreen")):
        mask = labels_arr == scale
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6, label=f"scale {scale}", color=color)
    ax.set_title("t-SNE of cross-scale embeddings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "tsne_embeddings.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved tsne_embeddings.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for d in (EMBEDDING_OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(LOGS_DIR / "step8.log")
    logger.info("Starting Step 8: Cross-Scale Spectral Consistency Learning (scales=%s)", SCALES)

    tokens_path = TOKEN_INPUT_DIR / "tokens.csv"
    if not tokens_path.exists():
        logger.error("Token file not found: %s", tokens_path)
        return

    token_df = pd.read_csv(tokens_path)
    feature_names_path = TOKEN_INPUT_DIR / "feature_names.json"
    if feature_names_path.exists():
        with feature_names_path.open() as f:
            feature_columns = json.load(f)
    else:
        feature_columns = [c for c in token_df.columns if c not in METADATA_COLUMNS]

    token_df = token_df.dropna(subset=feature_columns).reset_index(drop=True)
    n_tokens = len(token_df)
    if n_tokens < 4:
        logger.error("Not enough valid tokens (%d) to run cross-scale consistency learning.", n_tokens)
        return

    input_dim = len(feature_columns)
    logger.info("Loaded %d tokens with %d features across %d subjects.",
                n_tokens, input_dim, token_df["Subject_ID"].nunique())

    t0 = time.perf_counter()
    raw_views = build_scale_views(token_df, feature_columns, SCALES, logger)
    t1 = time.perf_counter()

    standardized_views = standardize_views(raw_views)
    model, loss_history = train_projection_head(standardized_views, SCALES, input_dim, logger)
    t2 = time.perf_counter()

    trained_embeddings = compute_all_embeddings(model, standardized_views, SCALES)
    t3 = time.perf_counter()

    before_sim = cross_scale_similarity(standardized_views, SCALES)
    after_sim = cross_scale_similarity(trained_embeddings, SCALES)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(before_sim, after_sim, alternative="less")

    sample_idx = np.random.default_rng(RANDOM_SEED).choice(
        n_tokens, size=min(RETRIEVAL_SAMPLE_SIZE, n_tokens), replace=False
    )
    hits_before = cross_scale_retrieval(standardized_views, SCALES, sample_idx)
    hits_after = cross_scale_retrieval(trained_embeddings, SCALES, sample_idx)
    retrieval_rows = [{
        "scale_pair": f"{a}-{b}",
        "n_tokens_sampled": len(sample_idx),
        "top1_retrieval_before": float(hits_before[(a, b)].mean()),
        "top1_retrieval_after": float(hits_after[(a, b)].mean()),
        "mcnemar_pvalue": mcnemar_exact(hits_before[(a, b)], hits_after[(a, b)]),
    } for a, b in hits_before]
    for row in retrieval_rows:
        logger.info("Cross-scale top-1 retrieval %s: %.1f%% -> %.1f%% (McNemar p=%.2e)",
                    row["scale_pair"], 100 * row["top1_retrieval_before"],
                    100 * row["top1_retrieval_after"], row["mcnemar_pvalue"])

    # --- Save embeddings ---
    for scale in SCALES:
        np.save(EMBEDDING_OUTPUT_DIR / f"embeddings_scale{scale}.npy", trained_embeddings[scale].astype(np.float32))
    token_df[["Subject_ID", "Patch_ID", "Hemisphere", "Atlas_Region"]].to_csv(
        EMBEDDING_OUTPUT_DIR / "token_index.csv", index=False
    )
    torch.save(model.state_dict(), EMBEDDING_OUTPUT_DIR / "projection_head.pt")
    logger.info("Saved trained embeddings and projection head -> %s", EMBEDDING_OUTPUT_DIR)

    # --- Metrics: training loss ---
    pd.DataFrame({"epoch": range(1, len(loss_history) + 1), "loss": loss_history}).to_csv(
        METRICS_DIR / "training_loss.csv", index=False
    )

    # --- Metrics: per-token cosine similarity ---
    similarity_df = token_df[["Subject_ID", "Patch_ID"]].copy()
    similarity_df["Before_Similarity"] = before_sim
    similarity_df["After_Similarity"] = after_sim
    similarity_df["Improvement"] = after_sim - before_sim
    similarity_df.to_csv(METRICS_DIR / "similarity.csv", index=False)

    # --- Metrics: validation ---
    # Primary: cross-scale top-1 retrieval before vs after (McNemar).
    pd.DataFrame(retrieval_rows).to_csv(METRICS_DIR / "validation_retrieval.csv", index=False)
    # Secondary, kept for reference: raw cosine before vs after (Wilcoxon) -
    # see cross_scale_retrieval() for why this one is not the right test.
    pd.DataFrame([{
        "n_tokens": n_tokens,
        "mean_before_similarity": float(np.mean(before_sim)),
        "std_before_similarity": float(np.std(before_sim)),
        "mean_after_similarity": float(np.mean(after_sim)),
        "std_after_similarity": float(np.std(after_sim)),
        "wilcoxon_statistic": float(wilcoxon_stat),
        "wilcoxon_pvalue": float(wilcoxon_p),
        "significant_improvement": bool(wilcoxon_p < 0.05),
    }]).to_csv(METRICS_DIR / "validation.csv", index=False)

    # --- Metrics: summary ---
    pd.DataFrame([{
        "n_subjects": int(token_df["Subject_ID"].nunique()),
        "n_tokens": n_tokens,
        "scales": str(SCALES),
        "embedding_dim": PROJECTION_OUTPUT_DIM,
        "temperature": TEMPERATURE,
        "epochs": N_EPOCHS,
        "final_training_loss": loss_history[-1] if loss_history else float("nan"),
    }]).to_csv(METRICS_DIR / "summary.csv", index=False)

    # --- Metrics: runtime ---
    pd.DataFrame([{
        "build_scale_views_seconds": round(t1 - t0, 3),
        "training_seconds": round(t2 - t1, 3),
        "embedding_seconds": round(t3 - t2, 3),
        "total_seconds": round(t3 - t0, 3),
    }]).to_csv(METRICS_DIR / "runtime.csv", index=False)

    # --- Visualizations ---
    plot_loss_curve(loss_history, FIGURES_DIR, logger)
    plot_similarity_matrix(trained_embeddings, SCALES, SIMILARITY_MATRIX_SAMPLE, FIGURES_DIR, logger)
    plot_tsne(trained_embeddings, SCALES, TSNE_SAMPLE_SIZE, FIGURES_DIR, logger)

    logger.info(
        "Step 8 complete. Tokens=%d | mean similarity before=%.4f after=%.4f | "
        "Wilcoxon p=%.5f | Total runtime=%.2f s",
        n_tokens, float(np.mean(before_sim)), float(np.mean(after_sim)), wilcoxon_p, t3 - t0,
    )


if __name__ == "__main__":
    main()
