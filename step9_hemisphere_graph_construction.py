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
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from pipeline_config import RUN_DIR

BASE_DIR = Path(RUN_DIR)
EMBEDDING_INPUT_DIR = BASE_DIR / "embedding_outputs"
# Step 8's token_index.csv (embedding_outputs/) carries Subject_ID/Patch_ID/
# Hemisphere/Atlas_Region but not spatial Coordinates. Step 7's tokens.csv
# still has them, in the same (Subject_ID, Patch_ID) key space, so it's
# used here purely as a coordinate lookup - every node/edge/embedding still
# comes from embedding_outputs/.
TOKEN_OUTPUT_DIR = BASE_DIR / "token_outputs"
GRAPH_OUTPUT_DIR = BASE_DIR / "graph_outputs"
FIGURES_DIR = BASE_DIR / "figures" / "step9"
METRICS_DIR = BASE_DIR / "metrics" / "step9"
LOGS_DIR = BASE_DIR / "logs"

SCALES = (24, 32, 48)  # must match Step 8's trained scales
HEMISPHERES = ("L", "R")
TOP_K = 5
DPI = 300


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step9_hemisphere_graph_construction")
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
# Loading
# ---------------------------------------------------------------------------
def parse_coordinates(coord_str: str) -> tuple[float, float, float]:
    parts = str(coord_str).strip("()").split(",")
    return float(parts[0]), float(parts[1]), float(parts[2])


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def load_node_table(
    embedding_dir: Path, token_output_dir: Path, scales: tuple[int, ...], logger: logging.Logger
) -> tuple[pd.DataFrame, np.ndarray]:
    index_df = pd.read_csv(embedding_dir / "token_index.csv")

    embed_stack = []
    for scale in scales:
        path = embedding_dir / f"embeddings_scale{scale}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {path}")
        embed_stack.append(np.load(path))

    # One canonical node embedding per token: the mean of its (already
    # cross-scale-consistency-trained) scale views, re-normalized. This is
    # the direct payoff of Step 8's contrastive training - the 3 views were
    # trained to agree, so averaging them is a legitimate, more robust
    # single representation rather than picking one scale arbitrarily.
    node_embeddings = l2_normalize(np.mean(np.stack(embed_stack, axis=0), axis=0))

    if len(index_df) != node_embeddings.shape[0]:
        raise ValueError(
            f"token_index.csv has {len(index_df)} rows but embeddings have "
            f"{node_embeddings.shape[0]} rows."
        )

    tokens_path = token_output_dir / "tokens.csv"
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Cannot find {tokens_path} - Step 9 needs Step 7's Coordinates column "
            f"for spatial edges (embedding_outputs/token_index.csv does not carry them)."
        )
    coords_df = pd.read_csv(tokens_path, usecols=["Subject_ID", "Patch_ID", "Coordinates"])

    merged = index_df.merge(coords_df, on=["Subject_ID", "Patch_ID"], how="left")
    if len(merged) != len(index_df):
        raise ValueError("Coordinate merge changed row count - duplicate (Subject_ID, Patch_ID) keys found.")

    if merged["Coordinates"].isna().any():
        logger.warning("%d nodes have no matching Coordinates entry in tokens.csv.",
                        int(merged["Coordinates"].isna().sum()))

    xyz = np.array([
        parse_coordinates(c) if pd.notna(c) else (np.nan, np.nan, np.nan)
        for c in merged["Coordinates"]
    ])
    merged["x"], merged["y"], merged["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    logger.info("Loaded %d nodes (embedding dim=%d, averaged over scales=%s).",
                len(merged), node_embeddings.shape[1], scales)
    return merged, node_embeddings


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_hemisphere_graph(
    node_df: pd.DataFrame, node_embeddings: np.ndarray, top_k: int, logger: logging.Logger
) -> Optional[dict]:
    coords = node_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    valid = ~np.isnan(coords).any(axis=1)
    if valid.sum() < 2:
        return None

    node_df = node_df.loc[valid].reset_index(drop=True)
    coords = coords[valid]
    embeddings = node_embeddings[valid]
    n_nodes = len(node_df)

    k = min(top_k, n_nodes - 1)
    if k < top_k:
        logger.warning("Only %d valid nodes; reducing top_k from %d to %d.", n_nodes, top_k, k)

    tree = cKDTree(coords)
    _, neighbor_idx = tree.query(coords, k=k + 1)
    neighbor_idx = np.atleast_2d(neighbor_idx)

    normed_embeddings = l2_normalize(embeddings)

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in neighbor_idx[i]:
            j = int(j)
            if j == i or graph.has_edge(i, j):
                continue
            weight = float(np.dot(normed_embeddings[i], normed_embeddings[j]))
            graph.add_edge(i, j, weight=weight)

    return {"graph": graph, "node_df": node_df, "embeddings": embeddings, "coords": coords}


def save_graph(subject_id: str, hemisphere: str, graph_data: dict, output_dir: Path) -> Path:
    graph = graph_data["graph"]
    node_df = graph_data["node_df"]
    embeddings = graph_data["embeddings"]

    sub_out_dir = output_dir / subject_id
    sub_out_dir.mkdir(parents=True, exist_ok=True)

    edge_list = list(graph.edges())
    if edge_list:
        edges = np.array(edge_list, dtype=np.int64)
        weights = np.array([graph[u][v]["weight"] for u, v in edge_list], dtype=np.float32)
        edge_index = np.concatenate([edges, edges[:, [1, 0]]], axis=0).T
        edge_weight = np.concatenate([weights, weights])
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)

    npz_path = sub_out_dir / f"{subject_id}_{hemisphere}_graph.npz"
    np.savez(
        npz_path,
        node_embeddings=embeddings.astype(np.float32),
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    nodes_csv_path = sub_out_dir / f"{subject_id}_{hemisphere}_nodes.csv"
    node_df[["Subject_ID", "Patch_ID", "Atlas_Region", "x", "y", "z"]].assign(
        Node_Index=range(len(node_df))
    ).to_csv(nodes_csv_path, index=False)

    return npz_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def compute_graph_metrics(graph: nx.Graph) -> dict:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    degrees = np.array([d for _, d in graph.degree()])
    weights = np.array([data["weight"] for _, _, data in graph.edges(data=True)])
    n_components = nx.number_connected_components(graph)
    largest_cc = max((len(c) for c in nx.connected_components(graph)), default=0)
    possible_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_connected_components": n_components,
        "largest_component_size": largest_cc,
        "largest_component_fraction": (largest_cc / n_nodes) if n_nodes else 0.0,
        "mean_degree": float(np.mean(degrees)) if degrees.size else 0.0,
        "min_degree": int(np.min(degrees)) if degrees.size else 0,
        "max_degree": int(np.max(degrees)) if degrees.size else 0,
        "median_degree": float(np.median(degrees)) if degrees.size else 0.0,
        "mean_edge_weight": float(np.mean(weights)) if weights.size else 0.0,
        "std_edge_weight": float(np.std(weights)) if weights.size else 0.0,
        "edge_density": (n_edges / possible_edges) if possible_edges else 0.0,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_graph_layout(vis_sample: dict, figures_dir: Path, logger: logging.Logger) -> None:
    hemispheres_present = [h for h in HEMISPHERES if h in vis_sample]
    if not hemispheres_present:
        logger.warning("No sample graph available for graph layout visualization.")
        return

    fig, axes = plt.subplots(1, len(hemispheres_present), figsize=(6.5 * len(hemispheres_present), 6))
    axes = np.atleast_1d(axes)

    for ax, hemisphere in zip(axes, hemispheres_present):
        graph_data = vis_sample[hemisphere]
        graph = graph_data["graph"]
        coords = graph_data["coords"]
        pos = {i: (coords[i, 0], coords[i, 1]) for i in range(len(coords))}
        color = "#1f77b4" if hemisphere == "L" else "#d62728"

        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3, width=0.6)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=25, node_color=color)
        ax.set_title(f"{vis_sample['subject_id']} - {hemisphere} hemisphere graph (top_k={TOP_K})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(figures_dir / "graph_layout.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved graph_layout.png")


def plot_adjacency_matrix(vis_sample: dict, figures_dir: Path, logger: logging.Logger) -> None:
    hemispheres_present = [h for h in HEMISPHERES if h in vis_sample]
    if not hemispheres_present:
        logger.warning("No sample graph available for adjacency matrix visualization.")
        return

    fig, axes = plt.subplots(1, len(hemispheres_present), figsize=(6.5 * len(hemispheres_present), 5.5))
    axes = np.atleast_1d(axes)

    for ax, hemisphere in zip(axes, hemispheres_present):
        graph = vis_sample[hemisphere]["graph"]
        adjacency = nx.to_numpy_array(graph, weight="weight")
        im = ax.imshow(adjacency, cmap="viridis")
        ax.set_title(f"{vis_sample['subject_id']} - {hemisphere} adjacency matrix")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(figures_dir / "adjacency_matrix.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved adjacency_matrix.png")


def plot_degree_histogram(degree_df: pd.DataFrame, figures_dir: Path, logger: logging.Logger) -> None:
    if degree_df.empty:
        logger.warning("No degree data available for degree histogram.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for hemisphere, color in zip(HEMISPHERES, ("steelblue", "darkorange")):
        values = degree_df.loc[degree_df["Hemisphere"] == hemisphere, "Degree"]
        if values.empty:
            continue
        bins = range(0, int(values.max()) + 2)
        ax.hist(values, bins=bins, alpha=0.6, label=hemisphere, color=color, edgecolor="black")

    ax.set_xlabel("Node degree")
    ax.set_ylabel("Node count")
    ax.set_title("Degree distribution across all hemisphere graphs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "degree_histogram.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved degree_histogram.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for d in (GRAPH_OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(LOGS_DIR / "step9.log")
    logger.info("Starting Step 9: Hemisphere Graph Construction (top_k=%d)", TOP_K)

    if not (EMBEDDING_INPUT_DIR / "token_index.csv").exists():
        logger.error("token_index.csv not found in %s", EMBEDDING_INPUT_DIR)
        return

    node_df, node_embeddings = load_node_table(EMBEDDING_INPUT_DIR, TOKEN_OUTPUT_DIR, SCALES, logger)

    subject_ids = sorted(node_df["Subject_ID"].unique())
    n_subjects = len(subject_ids)

    validation_rows: list[dict] = []
    degree_records: list[dict] = []
    vis_sample: Optional[dict] = None
    n_graphs_built = 0
    n_graphs_skipped = 0

    pipeline_start = time.perf_counter()

    for i, subject_id in enumerate(tqdm(subject_ids, desc="Subjects"), start=1):
        logger.info("Processing subject %d / %d (%s)", i, n_subjects, subject_id)
        subject_mask = node_df["Subject_ID"] == subject_id

        for hemisphere in HEMISPHERES:
            hemi_mask = subject_mask & (node_df["Hemisphere"] == hemisphere)
            sub_node_df = node_df.loc[hemi_mask].reset_index(drop=True)
            sub_indices = node_df.index[hemi_mask].to_numpy()
            sub_embeddings = node_embeddings[sub_indices]

            if len(sub_node_df) < 2:
                logger.warning("%s %s: fewer than 2 valid nodes, skipping graph.", subject_id, hemisphere)
                n_graphs_skipped += 1
                continue

            try:
                graph_data = build_hemisphere_graph(sub_node_df, sub_embeddings, TOP_K, logger)
            except (ValueError, RuntimeError) as exc:
                logger.error("%s %s: failed to build graph: %s", subject_id, hemisphere, exc)
                n_graphs_skipped += 1
                continue

            if graph_data is None:
                n_graphs_skipped += 1
                continue

            save_graph(subject_id, hemisphere, graph_data, GRAPH_OUTPUT_DIR)
            n_graphs_built += 1

            metrics = compute_graph_metrics(graph_data["graph"])
            metrics["Subject_ID"] = subject_id
            metrics["Hemisphere"] = hemisphere
            validation_rows.append(metrics)

            for _, degree in graph_data["graph"].degree():
                degree_records.append({"Subject_ID": subject_id, "Hemisphere": hemisphere, "Degree": degree})

            if vis_sample is None:
                vis_sample = {"subject_id": subject_id}
            if vis_sample.get("subject_id") == subject_id:
                vis_sample[hemisphere] = graph_data

    total_runtime = time.perf_counter() - pipeline_start

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(METRICS_DIR / "validation.csv", index=False)

    pd.DataFrame([{
        "n_subjects": n_subjects,
        "n_graphs_built": n_graphs_built,
        "n_graphs_skipped": n_graphs_skipped,
        "top_k": TOP_K,
        "mean_nodes_per_graph": float(validation_df["n_nodes"].mean()) if not validation_df.empty else 0.0,
        "mean_edges_per_graph": float(validation_df["n_edges"].mean()) if not validation_df.empty else 0.0,
        "mean_connected_components": float(validation_df["n_connected_components"].mean()) if not validation_df.empty else 0.0,
    }]).to_csv(METRICS_DIR / "summary.csv", index=False)

    degree_df = pd.DataFrame(degree_records)
    statistics_rows = []
    for hemisphere in HEMISPHERES:
        subset = degree_df.loc[degree_df["Hemisphere"] == hemisphere, "Degree"] if not degree_df.empty else pd.Series(dtype=float)
        if subset.empty:
            continue
        statistics_rows.append({
            "Hemisphere": hemisphere,
            "N_Nodes": len(subset),
            "Mean_Degree": float(subset.mean()),
            "Std_Degree": float(subset.std()),
            "Min_Degree": int(subset.min()),
            "Max_Degree": int(subset.max()),
            "Median_Degree": float(subset.median()),
        })
    pd.DataFrame(statistics_rows).to_csv(METRICS_DIR / "statistics.csv", index=False)

    pd.DataFrame([{"total_seconds": round(total_runtime, 3), "n_subjects": n_subjects}]).to_csv(
        METRICS_DIR / "runtime.csv", index=False
    )

    if vis_sample is not None:
        plot_graph_layout(vis_sample, FIGURES_DIR, logger)
        plot_adjacency_matrix(vis_sample, FIGURES_DIR, logger)
    else:
        logger.warning("No sample graph available for graph/adjacency visualizations.")
    plot_degree_histogram(degree_df, FIGURES_DIR, logger)

    logger.info(
        "Step 9 complete. Graphs built: %d | skipped: %d | Total runtime: %.2f s",
        n_graphs_built, n_graphs_skipped, total_runtime,
    )


if __name__ == "__main__":
    main()
