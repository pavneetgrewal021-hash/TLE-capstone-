from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from pipeline_config import RUN_DIR

BASE_DIR = Path(RUN_DIR)
EMBEDDING_INPUT_DIR = BASE_DIR / "embedding_outputs"
FUSION_OUTPUT_DIR = BASE_DIR / "fusion_outputs"
FIGURES_DIR = BASE_DIR / "figures" / "step11"
METRICS_DIR = BASE_DIR / "metrics" / "step11"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

GATE_HIDDEN_DIM = 16
LEARNING_RATE = 1e-3
N_EPOCHS = 100
RANDOM_SEED = 42
DPI = 300


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step11_adaptive_hemisphere_fusion")
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
# Loading: Step 10's graph-level (GAT) embeddings, paired by subject
# ---------------------------------------------------------------------------
def build_hemisphere_pairs(
    index_df: pd.DataFrame, embeddings: np.ndarray, logger: logging.Logger
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index_df = index_df.reset_index(drop=True)
    left_rows = index_df[index_df["Hemisphere"] == "L"]
    right_rows = index_df[index_df["Hemisphere"] == "R"]

    left_lookup = {row.Subject_ID: idx for idx, row in left_rows.iterrows()}
    right_lookup = {row.Subject_ID: idx for idx, row in right_rows.iterrows()}

    common_subjects = sorted(set(left_lookup) & set(right_lookup))
    missing = sorted((set(left_lookup) | set(right_lookup)) - set(common_subjects))
    if missing:
        logger.warning("%d subject(s) missing one hemisphere, excluded from fusion: %s", len(missing), missing)

    h_l = np.stack([embeddings[left_lookup[s]] for s in common_subjects])
    h_r = np.stack([embeddings[right_lookup[s]] for s in common_subjects])
    n_nodes_l = np.array([index_df.loc[left_lookup[s], "N_Nodes"] for s in common_subjects])
    n_nodes_r = np.array([index_df.loc[right_lookup[s], "N_Nodes"] for s in common_subjects])

    return common_subjects, h_l, h_r, n_nodes_l, n_nodes_r


# ---------------------------------------------------------------------------
# Adaptive fusion gate
# ---------------------------------------------------------------------------
class FusionGate(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_l: torch.Tensor, h_r: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([h_l, h_r, h_l - h_r], dim=-1)
        return torch.sigmoid(self.net(gate_input)).squeeze(-1)


class FusionDecoder(nn.Module):
    """Trains the gate: from the weighted-fusion vector alone, try to
    reconstruct both original hemisphere embeddings. A degenerate gate
    (e.g. always alpha=0.5) reconstructs both sides equally poorly for
    lateralized subjects, so gradient pressure pushes alpha toward
    genuinely subject-specific weighting - the only training signal
    available without diagnosis labels at this stage of the pipeline."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.decode_l = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
        self.decode_r = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decode_l(fused), self.decode_r(fused)


def train_adaptive_fusion(
    h_l: np.ndarray, h_r: np.ndarray, dim: int, logger: logging.Logger
) -> tuple[FusionGate, list[float]]:
    torch.manual_seed(RANDOM_SEED)
    gate = FusionGate(dim, GATE_HIDDEN_DIM)
    decoder = FusionDecoder(dim, GATE_HIDDEN_DIM)
    optimizer = torch.optim.Adam(list(gate.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

    x_l = torch.tensor(h_l, dtype=torch.float32)
    x_r = torch.tensor(h_r, dtype=torch.float32)

    loss_history: list[float] = []
    for epoch in tqdm(range(N_EPOCHS), desc="Training adaptive fusion gate"):
        gate.train()
        decoder.train()

        alpha = gate(x_l, x_r)
        fused = alpha.unsqueeze(-1) * x_l + (1 - alpha).unsqueeze(-1) * x_r
        recon_l, recon_r = decoder(fused)
        loss = nn.functional.mse_loss(recon_l, x_l) + nn.functional.mse_loss(recon_r, x_r)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))
        logger.info("Epoch %d/%d - fusion reconstruction loss: %.4f", epoch + 1, N_EPOCHS, loss_history[-1])

    return gate, loss_history


# ---------------------------------------------------------------------------
# Validation: compare average / concatenation / adaptive fusion by how well
# a LOOCV linear probe fit on each method's fused vector can reconstruct
# both original hemisphere embeddings - the same evaluation protocol
# applied uniformly to all three, so the comparison is fair even though
# only "adaptive" has learned parameters.
# ---------------------------------------------------------------------------
def evaluate_reconstruction_loocv(fused: np.ndarray, h_l: np.ndarray, h_r: np.ndarray) -> dict:
    n = fused.shape[0]
    if n < 3:
        return {"r2_left": float("nan"), "r2_right": float("nan"), "mean_r2": float("nan")}

    preds_l = np.zeros_like(h_l)
    preds_r = np.zeros_like(h_r)

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        reg_l = LinearRegression().fit(fused[train_idx], h_l[train_idx])
        reg_r = LinearRegression().fit(fused[train_idx], h_r[train_idx])
        preds_l[i] = reg_l.predict(fused[i:i + 1])[0]
        preds_r[i] = reg_r.predict(fused[i:i + 1])[0]

    r2_l = float(r2_score(h_l, preds_l))
    r2_r = float(r2_score(h_r, preds_r))
    return {"r2_left": r2_l, "r2_right": r2_r, "mean_r2": (r2_l + r2_r) / 2.0}


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_fusion_weights(alpha: np.ndarray, subject_ids: list[str], figures_dir: Path, logger: logging.Logger) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(alpha, bins=20, color="steelblue", edgecolor="black")
    axes[0].axvline(0.5, color="red", linestyle="--", label="Balanced (0.5)")
    axes[0].set_xlabel("Fusion weight alpha (favors Left as alpha -> 1)")
    axes[0].set_ylabel("Subject count")
    axes[0].set_title("Distribution of learned fusion weights")
    axes[0].legend()

    order = np.argsort(alpha)
    axes[1].bar(range(len(alpha)), alpha[order], color="steelblue")
    axes[1].axhline(0.5, color="red", linestyle="--")
    axes[1].set_xticks(range(len(alpha)))
    axes[1].set_xticklabels(np.array(subject_ids)[order], rotation=90, fontsize=6)
    axes[1].set_ylabel("alpha")
    axes[1].set_title("Per-subject fusion weight")

    fig.tight_layout()
    fig.savefig(figures_dir / "fusion_weights.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved fusion_weights.png")


def plot_embedding_comparison(
    fusion_vectors: dict[str, np.ndarray], alpha: np.ndarray, figures_dir: Path, logger: logging.Logger
) -> None:
    methods = list(fusion_vectors.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5.5))
    axes = np.atleast_1d(axes)

    sc = None
    for ax, method in zip(axes, methods):
        emb = fusion_vectors[method]
        coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(emb) if emb.shape[1] > 2 else emb
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=alpha, cmap="coolwarm", vmin=0, vmax=1, s=40)
        ax.set_title(f"{method} (dim={emb.shape[1]})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    if sc is not None:
        fig.colorbar(sc, ax=axes[-1], label="fusion weight alpha")

    fig.tight_layout()
    fig.savefig(figures_dir / "embedding_comparison.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved embedding_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for d in (FUSION_OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(LOGS_DIR / "step11.log")
    logger.info("Starting Step 11: Adaptive Hemisphere Fusion")

    index_path = EMBEDDING_INPUT_DIR / "graph_embedding_index.csv"
    embed_path = EMBEDDING_INPUT_DIR / "graph_embeddings_gat.npy"
    if not index_path.exists() or not embed_path.exists():
        logger.error(
            "Missing graph embeddings in %s (expected graph_embedding_index.csv + "
            "graph_embeddings_gat.npy from Step 10).", EMBEDDING_INPUT_DIR
        )
        return

    index_df = pd.read_csv(index_path)
    embeddings = np.load(embed_path)
    if len(index_df) != embeddings.shape[0]:
        logger.error("graph_embedding_index.csv row count does not match graph_embeddings_gat.npy.")
        return

    t0 = time.perf_counter()
    subject_ids, h_l, h_r, n_nodes_l, n_nodes_r = build_hemisphere_pairs(index_df, embeddings, logger)
    if len(subject_ids) < 3:
        logger.error("Not enough subjects with both hemispheres present (%d) to run fusion.", len(subject_ids))
        return
    t1 = time.perf_counter()

    dim = h_l.shape[1]
    logger.info("Paired %d subjects with both hemispheres (embedding dim=%d).", len(subject_ids), dim)

    average_fusion = (h_l + h_r) / 2.0
    concat_fusion = np.concatenate([h_l, h_r], axis=1)

    gate, loss_history = train_adaptive_fusion(h_l, h_r, dim, logger)
    t2 = time.perf_counter()

    gate.eval()
    with torch.no_grad():
        alpha = gate(torch.tensor(h_l, dtype=torch.float32), torch.tensor(h_r, dtype=torch.float32)).numpy()
    asymmetry = h_l - h_r
    weighted = alpha[:, None] * h_l + (1 - alpha[:, None]) * h_r
    adaptive_fusion = np.concatenate([weighted, asymmetry], axis=1)
    t3 = time.perf_counter()

    fusion_vectors = {"average": average_fusion, "concatenation": concat_fusion, "adaptive": adaptive_fusion}

    comparison_rows = []
    for method, vec in fusion_vectors.items():
        result = evaluate_reconstruction_loocv(vec, h_l, h_r)
        result["Method"] = method
        result["Embedding_Dim"] = vec.shape[1]
        comparison_rows.append(result)
    t4 = time.perf_counter()

    # --- Save outputs ---
    np.save(FUSION_OUTPUT_DIR / "fusion_embeddings_average.npy", average_fusion.astype(np.float32))
    np.save(FUSION_OUTPUT_DIR / "fusion_embeddings_concat.npy", concat_fusion.astype(np.float32))
    np.save(FUSION_OUTPUT_DIR / "fusion_embeddings_adaptive.npy", adaptive_fusion.astype(np.float32))
    np.save(FUSION_OUTPUT_DIR / "asymmetry_vectors.npy", asymmetry.astype(np.float32))

    pd.DataFrame({
        "Subject_ID": subject_ids,
        "Alpha": alpha,
        "N_Nodes_L": n_nodes_l,
        "N_Nodes_R": n_nodes_r,
    }).to_csv(FUSION_OUTPUT_DIR / "alpha_weights.csv", index=False)
    logger.info("Saved fusion embeddings -> %s", FUSION_OUTPUT_DIR)

    torch.save(gate.state_dict(), MODELS_DIR / "step11_fusion_gate.pt")
    logger.info("Saved trained fusion gate -> %s", MODELS_DIR)

    # --- Metrics ---
    pd.DataFrame({"epoch": range(1, len(loss_history) + 1), "loss": loss_history}).to_csv(
        METRICS_DIR / "training_loss.csv", index=False
    )
    pd.DataFrame(comparison_rows)[["Method", "Embedding_Dim", "r2_left", "r2_right", "mean_r2"]].to_csv(
        METRICS_DIR / "validation.csv", index=False
    )
    pd.DataFrame([{
        "n_subjects": len(subject_ids),
        "embedding_dim_per_hemisphere": dim,
        "alpha_mean": float(np.mean(alpha)),
        "alpha_std": float(np.std(alpha)),
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
        "alpha_median": float(np.median(alpha)),
    }]).to_csv(METRICS_DIR / "statistics.csv", index=False)

    best_method = max(comparison_rows, key=lambda r: r["mean_r2"])["Method"]
    pd.DataFrame([{
        "n_subjects": len(subject_ids),
        "embedding_dim_per_hemisphere": dim,
        "final_training_loss": loss_history[-1] if loss_history else float("nan"),
        "best_reconstruction_method": best_method,
    }]).to_csv(METRICS_DIR / "summary.csv", index=False)

    pd.DataFrame([{
        "load_seconds": round(t1 - t0, 3),
        "gate_training_seconds": round(t2 - t1, 3),
        "fusion_compute_seconds": round(t3 - t2, 3),
        "loocv_comparison_seconds": round(t4 - t3, 3),
        "total_seconds": round(t4 - t0, 3),
    }]).to_csv(METRICS_DIR / "runtime.csv", index=False)

    # --- Visualizations ---
    plot_fusion_weights(alpha, subject_ids, FIGURES_DIR, logger)
    plot_embedding_comparison(fusion_vectors, alpha, FIGURES_DIR, logger)

    logger.info(
        "Step 11 complete. Subjects=%d | mean alpha=%.3f | best method (mean R2)=%s | Total runtime=%.2f s",
        len(subject_ids), float(np.mean(alpha)), best_method, t4 - t0,
    )


if __name__ == "__main__":
    main()
