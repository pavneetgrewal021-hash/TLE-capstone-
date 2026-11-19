from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from pipeline_config import RUN_DIR

BASE_DIR = Path(RUN_DIR)
GRAPH_INPUT_DIR = BASE_DIR / "graph_outputs"
EMBEDDING_OUTPUT_DIR = BASE_DIR / "embedding_outputs"
FIGURES_DIR = BASE_DIR / "figures" / "step10"
METRICS_DIR = BASE_DIR / "metrics" / "step10"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

GAT_HIDDEN_DIM = 16
GAT_OUTPUT_DIM = 16
GAT_HEADS = 4
GAT_DROPOUT = 0.1
PCA_OUTPUT_DIM = 16

TEMPERATURE = 0.07
LEARNING_RATE = 1e-3
N_EPOCHS = 40
LOOCV_EPOCHS = 15
EDGE_DROPOUT = 0.2
FEATURE_MASK_PROB = 0.1
RANDOM_SEED = 42
DPI = 300


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step10_graph_representation_learning")
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
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class GraphSample:
    subject_id: str
    hemisphere: str
    node_features: np.ndarray
    edge_index: np.ndarray
    n_nodes: int


def discover_graphs(graph_dir: Path) -> list[tuple[str, str, Path]]:
    entries: list[tuple[str, str, Path]] = []
    if not graph_dir.exists():
        return entries
    for subject_dir in sorted(graph_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        for npz_path in sorted(subject_dir.glob("*_graph.npz")):
            subject_id, hemisphere, _ = npz_path.stem.rsplit("_", 2)
            entries.append((subject_id, hemisphere, npz_path))
    return entries


def load_graph(npz_path: Path, subject_id: str, hemisphere: str) -> GraphSample:
    data = np.load(npz_path)
    node_features = data["node_embeddings"].astype(np.float32)
    edge_index = data["edge_index"].astype(np.int64)
    return GraphSample(subject_id, hemisphere, node_features, edge_index, node_features.shape[0])


def load_sample_coordinates(graph_dir: Path, subject_id: str, hemisphere: str) -> Optional[np.ndarray]:
    nodes_csv = graph_dir / subject_id / f"{subject_id}_{hemisphere}_nodes.csv"
    if not nodes_csv.exists():
        return None
    df = pd.read_csv(nodes_csv).sort_values("Node_Index")
    return df[["x", "y"]].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Graph Attention Network (implemented directly on edge_index, no
# torch_geometric dependency, to keep this a single portable file)
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        h = self.W(x).view(n, self.n_heads, self.out_dim)

        if edge_index.shape[1] == 0:
            return h.reshape(n, self.n_heads * self.out_dim), torch.zeros(0, self.n_heads, device=x.device), \
                torch.zeros(0, dtype=torch.long, device=x.device)

        src, dst = edge_index[0], edge_index[1]
        alpha_src = (h * self.attn_src).sum(dim=-1)
        alpha_dst = (h * self.attn_dst).sum(dim=-1)
        e = self.leaky_relu(alpha_src[src] + alpha_dst[dst])

        e_exp = torch.exp(e - e.max())
        denom = torch.zeros(n, self.n_heads, device=x.device)
        denom.index_add_(0, dst, e_exp)
        denom = denom[dst].clamp(min=1e-9)
        attn = self.dropout(e_exp / denom)

        weighted = h[src] * attn.unsqueeze(-1)
        out = torch.zeros(n, self.n_heads, self.out_dim, device=x.device)
        out.index_add_(0, dst, weighted)

        return out.reshape(n, self.n_heads * self.out_dim), attn, dst


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = GraphAttentionLayer(in_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.layer2 = GraphAttentionLayer(hidden_dim * n_heads, out_dim, n_heads=1, dropout=dropout)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, _, _ = self.layer1(x, edge_index)
        h = self.elu(h)
        h, attn, dst = self.layer2(h, edge_index)
        return h, attn, dst


def attention_pool(
    node_embeddings: torch.Tensor, attn: torch.Tensor, dst: torch.Tensor, n_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    device = node_embeddings.device
    if attn.numel() == 0:
        node_importance = torch.ones(n_nodes, device=device) / max(n_nodes, 1)
        return node_embeddings.mean(dim=0), node_importance

    edge_importance = attn.mean(dim=-1) if attn.dim() > 1 else attn
    node_importance = torch.zeros(n_nodes, device=device)
    node_importance.index_add_(0, dst, edge_importance)
    total = node_importance.sum()
    node_importance = node_importance / total if total > 0 else torch.ones(n_nodes, device=device) / max(n_nodes, 1)
    graph_embedding = (node_embeddings * node_importance.unsqueeze(-1)).sum(dim=0)
    return graph_embedding, node_importance


def info_nce_loss(embeddings: torch.Tensor, anchor_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = embeddings @ embeddings.T / temperature
    n = sim.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    sim = sim.masked_fill(self_mask, -1e9)

    positive_mask = (anchor_ids.unsqueeze(0) == anchor_ids.unsqueeze(1)) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = positive_mask.sum(dim=1).clamp(min=1)
    positive_log_prob = (log_prob * positive_mask).sum(dim=1) / pos_counts
    return -positive_log_prob.mean()


def augment_graph(
    sample: GraphSample, edge_drop_prob: float, feature_mask_prob: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    edge_index = sample.edge_index
    n_edges = edge_index.shape[1]
    if n_edges > 0:
        keep_mask = rng.random(n_edges) > edge_drop_prob
        if not keep_mask.any():
            keep_mask[:] = True
        edge_index = edge_index[:, keep_mask]

    features = sample.node_features.copy()
    feat_mask = rng.random(features.shape[1]) < feature_mask_prob
    features[:, feat_mask] = 0.0
    return features, edge_index


def train_gat(
    samples: list[GraphSample], input_dim: int, logger: logging.Logger, n_epochs: int, verbose: bool
) -> tuple[GATEncoder, list[float]]:
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")
    model = GATEncoder(input_dim, GAT_HIDDEN_DIM, GAT_OUTPUT_DIM, n_heads=GAT_HEADS, dropout=GAT_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(RANDOM_SEED)

    loss_history: list[float] = []

    for epoch in tqdm(range(n_epochs), desc="Training GAT", leave=verbose):
        model.train()
        view1_embeds, view2_embeds = [], []

        for sample in samples:
            f1, e1 = augment_graph(sample, EDGE_DROPOUT, FEATURE_MASK_PROB, rng)
            f2, e2 = augment_graph(sample, EDGE_DROPOUT, FEATURE_MASK_PROB, rng)

            x1 = torch.tensor(f1, dtype=torch.float32, device=device)
            x2 = torch.tensor(f2, dtype=torch.float32, device=device)
            ei1 = torch.tensor(e1, dtype=torch.long, device=device)
            ei2 = torch.tensor(e2, dtype=torch.long, device=device)

            n1, a1, d1 = model(x1, ei1)
            n2, a2, d2 = model(x2, ei2)
            g1, _ = attention_pool(n1, a1, d1, sample.n_nodes)
            g2, _ = attention_pool(n2, a2, d2, sample.n_nodes)

            view1_embeds.append(g1)
            view2_embeds.append(g2)

        embeddings = torch.cat([
            nn.functional.normalize(torch.stack(view1_embeds), dim=-1),
            nn.functional.normalize(torch.stack(view2_embeds), dim=-1),
        ], dim=0)
        n_graphs = len(samples)
        anchor_ids = torch.tensor(np.tile(np.arange(n_graphs), 2), dtype=torch.long, device=device)

        loss = info_nce_loss(embeddings, anchor_ids, TEMPERATURE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))
        if verbose:
            logger.info("Epoch %d/%d - GraphCL InfoNCE loss: %.4f", epoch + 1, n_epochs, loss_history[-1])

    return model, loss_history


def embed_graph(model: GATEncoder, sample: GraphSample, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x = torch.tensor(sample.node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(sample.edge_index, dtype=torch.long, device=device)
    with torch.no_grad():
        node_embeddings, attn, dst = model(x, edge_index)
        graph_embedding, node_importance = attention_pool(node_embeddings, attn, dst, sample.n_nodes)
    return graph_embedding.cpu().numpy(), node_importance.cpu().numpy()


# ---------------------------------------------------------------------------
# Baseline PCA
# ---------------------------------------------------------------------------
def compute_baseline_features(sample: GraphSample) -> np.ndarray:
    mean_pool = sample.node_features.mean(axis=0)
    max_pool = sample.node_features.max(axis=0)

    graph = nx.Graph()
    graph.add_nodes_from(range(sample.n_nodes))
    graph.add_edges_from(sample.edge_index.T.tolist())

    degrees = np.array([d for _, d in graph.degree()])
    mean_degree = float(np.mean(degrees)) if degrees.size else 0.0
    clustering = float(np.mean(list(nx.clustering(graph).values()))) if sample.n_nodes > 0 else 0.0
    possible_edges = sample.n_nodes * (sample.n_nodes - 1) / 2 if sample.n_nodes > 1 else 0
    density = graph.number_of_edges() / possible_edges if possible_edges else 0.0

    scalar_features = np.array([mean_degree, clustering, density, float(sample.n_nodes)])
    return np.concatenate([mean_pool, max_pool, scalar_features]).astype(np.float64)


def fit_pca_baseline(samples: list[GraphSample], n_components: int) -> tuple[np.ndarray, PCA, StandardScaler]:
    raw = np.stack([compute_baseline_features(s) for s in samples])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    n_components = max(1, min(n_components, scaled.shape[0], scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    embeddings = pca.fit_transform(scaled)
    return embeddings, pca, scaler


# ---------------------------------------------------------------------------
# LOOCV: retrain excluding one subject, compare its held-out embedding to
# the one produced by the full-data model. No labels exist yet at this
# stage of the pipeline, so this validates representation STABILITY
# (does leaving a subject out change its own embedding much) rather than
# classification accuracy.
# ---------------------------------------------------------------------------
def cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def run_loocv(
    samples: list[GraphSample], input_dim: int,
    full_gat_lookup: dict[tuple[str, str], np.ndarray],
    full_pca_lookup: dict[tuple[str, str], np.ndarray],
    pca_n_components: int, logger: logging.Logger,
) -> pd.DataFrame:
    subject_ids = sorted({s.subject_id for s in samples})
    rows: list[dict] = []

    for subject_id in tqdm(subject_ids, desc="LOOCV"):
        train_samples = [s for s in samples if s.subject_id != subject_id]
        held_out_samples = [s for s in samples if s.subject_id == subject_id]
        if len(train_samples) < 2 or not held_out_samples:
            logger.warning("LOOCV: skipping %s (not enough remaining graphs to train on).", subject_id)
            continue

        loocv_model, loocv_loss = train_gat(train_samples, input_dim, logger, n_epochs=LOOCV_EPOCHS, verbose=False)

        train_raw = np.stack([compute_baseline_features(s) for s in train_samples])
        scaler = StandardScaler().fit(train_raw)
        n_comp = max(1, min(pca_n_components, train_raw.shape[0], train_raw.shape[1]))
        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit(scaler.transform(train_raw))

        for sample in held_out_samples:
            loocv_gat_embed, _ = embed_graph(loocv_model, sample, torch.device("cpu"))
            gat_cos = cosine_similarity_vec(loocv_gat_embed, full_gat_lookup[(sample.subject_id, sample.hemisphere)])

            baseline_feat = compute_baseline_features(sample).reshape(1, -1)
            loocv_pca_embed = pca.transform(scaler.transform(baseline_feat))[0]
            pca_cos = cosine_similarity_vec(loocv_pca_embed, full_pca_lookup[(sample.subject_id, sample.hemisphere)])

            rows.append({
                "Subject_ID": sample.subject_id,
                "Hemisphere": sample.hemisphere,
                "GAT_Cosine_Similarity": gat_cos,
                "PCA_Cosine_Similarity": pca_cos,
            })

        logger.info("LOOCV fold %s done (final loocv training loss=%.4f).",
                     subject_id, loocv_loss[-1] if loocv_loss else float("nan"))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_attention_map(
    sample: GraphSample, node_importance: np.ndarray, coords: Optional[np.ndarray],
    figures_dir: Path, logger: logging.Logger,
) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(sample.n_nodes))
    graph.add_edges_from(sample.edge_index.T.tolist())

    if coords is None:
        pos_dict = nx.spring_layout(graph, seed=RANDOM_SEED)
        coords = np.array([pos_dict[i] for i in range(sample.n_nodes)])

    pos = {i: (coords[i, 0], coords[i, 1]) for i in range(sample.n_nodes)}
    sizes = 60 + 600 * (node_importance / (node_importance.max() + 1e-9))

    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25, width=0.5)
    nodes = nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes, node_color=node_importance, cmap="inferno")
    fig.colorbar(nodes, ax=ax, label="attention-derived node importance")
    ax.set_title(f"{sample.subject_id} - {sample.hemisphere} GAT attention map")
    fig.tight_layout()
    fig.savefig(figures_dir / "attention_map.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved attention_map.png")


def plot_training_curve(loss_history: list[float], figures_dir: Path, logger: logging.Logger) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GraphCL InfoNCE loss")
    ax.set_title("GAT graph-level contrastive training curve")
    fig.tight_layout()
    fig.savefig(figures_dir / "training_curve.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved training_curve.png")


def plot_embedding_visualization(
    gat_embeddings: np.ndarray, pca_embeddings: np.ndarray, index_df: pd.DataFrame,
    figures_dir: Path, logger: logging.Logger,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    hemi_codes, hemi_labels = pd.factorize(index_df["Hemisphere"])

    for ax, embeddings, title in zip(axes, (gat_embeddings, pca_embeddings), ("GAT embeddings", "PCA baseline embeddings")):
        if embeddings.shape[1] > 2:
            coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(embeddings)
        else:
            coords = embeddings
        ax.scatter(coords[:, 0], coords[:, 1], c=hemi_codes, cmap="tab10", s=30, alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    denom = max(len(hemi_labels) - 1, 1)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=plt.cm.tab10(i / denom))
               for i in range(len(hemi_labels))]
    axes[-1].legend(handles, list(hemi_labels), fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(figures_dir / "embedding_visualization.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved embedding_visualization.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for d in (EMBEDDING_OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(LOGS_DIR / "step10.log")
    logger.info("Starting Step 10: Graph Representation Learning")

    entries = discover_graphs(GRAPH_INPUT_DIR)
    if not entries:
        logger.error("No graphs found in %s", GRAPH_INPUT_DIR)
        return

    t0 = time.perf_counter()
    samples: list[GraphSample] = []
    for subject_id, hemisphere, npz_path in tqdm(entries, desc="Loading graphs"):
        try:
            samples.append(load_graph(npz_path, subject_id, hemisphere))
        except (OSError, KeyError, ValueError) as exc:
            logger.error("Failed to load %s: %s", npz_path, exc)

    if len(samples) < 2:
        logger.error("Not enough valid graphs (%d) to run representation learning.", len(samples))
        return

    input_dim = samples[0].node_features.shape[1]
    logger.info("Loaded %d graphs (node feature dim=%d).", len(samples), input_dim)
    t1 = time.perf_counter()

    pca_embeddings, pca_model, pca_scaler = fit_pca_baseline(samples, PCA_OUTPUT_DIM)
    t2 = time.perf_counter()

    model, loss_history = train_gat(samples, input_dim, logger, n_epochs=N_EPOCHS, verbose=True)
    t3 = time.perf_counter()

    gat_embeddings = np.zeros((len(samples), GAT_OUTPUT_DIM), dtype=np.float32)
    node_importance_by_sample: dict[tuple[str, str], np.ndarray] = {}
    for i, sample in enumerate(samples):
        embed, importance = embed_graph(model, sample, torch.device("cpu"))
        gat_embeddings[i] = embed
        node_importance_by_sample[(sample.subject_id, sample.hemisphere)] = importance
    t4 = time.perf_counter()

    full_gat_lookup = {(s.subject_id, s.hemisphere): gat_embeddings[i] for i, s in enumerate(samples)}
    full_pca_lookup = {(s.subject_id, s.hemisphere): pca_embeddings[i] for i, s in enumerate(samples)}

    # The leave-one-out check retrains the GAT once per subject (2.6 h on the
    # laptop). It validates stability only and does not change the
    # embeddings, so experiment runs can skip it with TLE_SKIP_STABILITY=1.
    if os.environ.get("TLE_SKIP_STABILITY") == "1":
        logger.info("Skipping the leave-one-out stability check (TLE_SKIP_STABILITY=1).")
        loocv_df = pd.DataFrame(columns=["Subject_ID", "Hemisphere", "GAT_Cosine_Similarity", "PCA_Cosine_Similarity"])
    else:
        loocv_df = run_loocv(samples, input_dim, full_gat_lookup, full_pca_lookup, pca_embeddings.shape[1], logger)
    t5 = time.perf_counter()

    # --- Save embeddings ---
    index_df = pd.DataFrame([
        {"Subject_ID": s.subject_id, "Hemisphere": s.hemisphere, "N_Nodes": s.n_nodes} for s in samples
    ])
    index_df.to_csv(EMBEDDING_OUTPUT_DIR / "graph_embedding_index.csv", index=False)
    np.save(EMBEDDING_OUTPUT_DIR / "graph_embeddings_gat.npy", gat_embeddings)
    np.save(EMBEDDING_OUTPUT_DIR / "graph_embeddings_pca.npy", pca_embeddings.astype(np.float32))
    logger.info("Saved graph-level embeddings -> %s", EMBEDDING_OUTPUT_DIR)

    torch.save(model.state_dict(), MODELS_DIR / "step10_gat_encoder.pt")
    with (MODELS_DIR / "step10_pca_baseline.pkl").open("wb") as f:
        pickle.dump({"pca": pca_model, "scaler": pca_scaler}, f)
    logger.info("Saved trained models -> %s", MODELS_DIR)

    # --- Metrics ---
    pd.DataFrame({"epoch": range(1, len(loss_history) + 1), "loss": loss_history}).to_csv(
        METRICS_DIR / "training_loss.csv", index=False
    )
    loocv_df.to_csv(METRICS_DIR / "validation.csv", index=False)

    statistics_rows = []
    for name, emb in (("GAT", gat_embeddings), ("PCA", pca_embeddings)):
        flat = emb.ravel()
        statistics_rows.append({
            "Method": name,
            "N_Graphs": emb.shape[0],
            "Embedding_Dim": emb.shape[1],
            "Mean": float(np.mean(flat)),
            "Std": float(np.std(flat)),
            "Min": float(np.min(flat)),
            "Max": float(np.max(flat)),
        })
    pd.DataFrame(statistics_rows).to_csv(METRICS_DIR / "statistics.csv", index=False)

    pd.DataFrame([{
        "n_graphs": len(samples),
        "n_subjects": int(index_df["Subject_ID"].nunique()),
        "gat_output_dim": GAT_OUTPUT_DIM,
        "pca_output_dim": int(pca_embeddings.shape[1]),
        "final_training_loss": loss_history[-1] if loss_history else float("nan"),
        "mean_loocv_cosine_gat": float(loocv_df["GAT_Cosine_Similarity"].mean()) if not loocv_df.empty else float("nan"),
        "mean_loocv_cosine_pca": float(loocv_df["PCA_Cosine_Similarity"].mean()) if not loocv_df.empty else float("nan"),
    }]).to_csv(METRICS_DIR / "summary.csv", index=False)

    pd.DataFrame([{
        "load_graphs_seconds": round(t1 - t0, 3),
        "pca_baseline_seconds": round(t2 - t1, 3),
        "gat_training_seconds": round(t3 - t2, 3),
        "embedding_seconds": round(t4 - t3, 3),
        "loocv_seconds": round(t5 - t4, 3),
        "total_seconds": round(t5 - t0, 3),
    }]).to_csv(METRICS_DIR / "runtime.csv", index=False)

    # --- Visualizations ---
    sample0 = samples[0]
    coords0 = load_sample_coordinates(GRAPH_INPUT_DIR, sample0.subject_id, sample0.hemisphere)
    importance0 = node_importance_by_sample[(sample0.subject_id, sample0.hemisphere)]
    plot_attention_map(sample0, importance0, coords0, FIGURES_DIR, logger)
    plot_training_curve(loss_history, FIGURES_DIR, logger)
    plot_embedding_visualization(gat_embeddings, pca_embeddings, index_df, FIGURES_DIR, logger)

    logger.info(
        "Step 10 complete. Graphs=%d | GAT final loss=%.4f | "
        "LOOCV mean cosine (GAT/PCA)=%.4f/%.4f | Total runtime=%.2f s",
        len(samples), loss_history[-1] if loss_history else float("nan"),
        float(loocv_df["GAT_Cosine_Similarity"].mean()) if not loocv_df.empty else float("nan"),
        float(loocv_df["PCA_Cosine_Similarity"].mean()) if not loocv_df.empty else float("nan"),
        t5 - t0,
    )


if __name__ == "__main__":
    main()
