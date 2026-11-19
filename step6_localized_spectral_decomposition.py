from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pywt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from pipeline_config import RUN_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(RUN_DIR)
PATCH_INPUT_DIR = BASE_DIR / "patch_outputs"
SPECTRAL_OUTPUT_DIR = BASE_DIR / "spectral_outputs"
FIGURES_DIR = BASE_DIR / "figures" / "step6"
METRICS_DIR = BASE_DIR / "metrics" / "step6"
LOGS_DIR = BASE_DIR / "logs"

WAVELET = "db4"
DECOMP_LEVEL = 3
PAD_MODE = "symmetric"

# Perfect-reconstruction tolerance. db4 + symmetric mode is designed for
# near machine-precision round-trip, so this is intentionally tight. This
# is the ONLY pass/fail gate below - it's a direct, reliable measure of
# "did the transform preserve all information," mirroring how Step 1 uses
# MI alone as its sole registration gate.
RECON_MSE_THRESHOLD = 1e-6

# NOTE ON ENERGY CONSERVATION: coefficient energy (sum of squares) is NOT
# expected to equal original patch energy for this configuration - a
# quick empirical check confirmed the ratio sits around ~2.4x, not ~1.0.
# Cause: pywt.coeffs_to_array pads each subband so they tile into one
# regular array (32^3 patch -> a ~52^3 packed array here, for db4/level 3/
# symmetric mode), and symmetric boundary extension is redundant, not
# orthogonal - so Parseval's theorem doesn't hold exactly, independent of
# whether the transform is correct. This mirrors the Step 1 Dice lesson:
# a metric that doesn't behave the "textbook" way isn't automatically a
# bug. What DOES matter, and is checked below, is that the ratio is
# consistent (low variance) across patches - a big per-patch swing would
# flag a real anomaly even though "close to 1.0" isn't the right target.
ENERGY_RATIO_CV_TOLERANCE = 0.10

DPI = 300
CHANNEL_NAMES = ("T1", "FLAIR")

# DISK: the full packed coefficient array is ~52^3 floats per patch per
# channel (~400 MB per subject, 30 GB for 75 subjects in the first run),
# but Step 7 only ever reduces it to these 7 summary statistics. So they
# are computed here, while the coefficients are already in memory, and
# only the (n_patches, 7) feature arrays are saved. Step 7's output is
# unchanged - same function, same inputs, just run one step earlier.
FEATURE_NAMES = ("Energy", "Entropy", "Mean", "Variance", "Skewness", "Kurtosis", "MAD")
ENTROPY_BINS = 64


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step6_spectral_decomposition")
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
def discover_subjects(patch_dir: Path) -> list[Path]:
    if not patch_dir.exists():
        return []
    return sorted(
        p for p in patch_dir.iterdir()
        if p.is_dir() and (p / "patches.npy").exists()
    )


# ---------------------------------------------------------------------------
# Running (streaming) statistics, so dataset-wide coefficient stats don't
# require holding every subject's coefficients in memory at once.
# ---------------------------------------------------------------------------
class RunningStats:
    def __init__(self) -> None:
        self.count: int = 0
        self.total: float = 0.0
        self.total_sq: float = 0.0
        self.min_val: float = np.inf
        self.max_val: float = -np.inf

    def update(self, arr: np.ndarray) -> None:
        flat = arr.astype(np.float64).ravel()
        if flat.size == 0:
            return
        self.count += int(flat.size)
        self.total += float(flat.sum())
        self.total_sq += float(np.sum(flat ** 2))
        self.min_val = min(self.min_val, float(flat.min()))
        self.max_val = max(self.max_val, float(flat.max()))

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def std(self) -> float:
        if self.count == 0:
            return 0.0
        m = self.mean()
        variance = max(self.total_sq / self.count - m ** 2, 0.0)
        return float(np.sqrt(variance))


# ---------------------------------------------------------------------------
# Core spectral decomposition
# ---------------------------------------------------------------------------
def decompose_patch(
    patch: np.ndarray, wavelet: str, level: int, mode: str
) -> tuple[np.ndarray, list, list]:
    """3D DWT of one patch. Returns the flattened coefficient array (via
    pywt.coeffs_to_array, so every patch of the same shape/wavelet/level
    produces an identically-shaped array - trivial to stack across
    patches), the slice map needed to invert coeffs_to_array, and the raw
    per-level coeffs list (used for the power-spectrum energy breakdown)."""
    patch64 = patch.astype(np.float64)
    coeffs = pywt.wavedecn(patch64, wavelet=wavelet, level=level, mode=mode)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return coeff_arr, coeff_slices, coeffs


def reconstruct_patch(coeff_arr: np.ndarray, coeff_slices: list, wavelet: str, mode: str) -> np.ndarray:
    coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format="wavedecn")
    return pywt.waverecn(coeffs, wavelet=wavelet, mode=mode)


def compute_energy_ratio(original: np.ndarray, coeff_arr: np.ndarray) -> float:
    energy_original = float(np.sum(original.astype(np.float64) ** 2))
    energy_coeffs = float(np.sum(coeff_arr.astype(np.float64) ** 2))
    if energy_original == 0:
        return 1.0
    return energy_coeffs / energy_original


def compute_entropy(values: np.ndarray, bins: int) -> float:
    hist, _ = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def compute_token_features(coeff_array: np.ndarray) -> list[float]:
    """7-value summary of one patch's wavelet coefficient array for one
    channel, in FEATURE_NAMES order (moved here from Step 7)."""
    values = coeff_array.astype(np.float64).ravel()
    std = float(np.std(values))
    median = float(np.median(values))
    return [
        float(np.sum(values ** 2)),
        compute_entropy(values, ENTROPY_BINS),
        float(np.mean(values)),
        float(np.var(values)),
        float(stats.skew(values)) if std > 0 else 0.0,
        float(stats.kurtosis(values)) if std > 0 else 0.0,
        float(np.median(np.abs(values - median))),
    ]


def per_level_energy(coeffs: list) -> dict[str, float]:
    """coeffs is pywt.wavedecn's output: [cA_n, {level n details}, ...,
    {level 1 details}]. Returns summed-squared energy per frequency band:
    the coarsest approximation plus each level's combined detail energy."""
    n_levels = len(coeffs) - 1
    energies: dict[str, float] = {
        "approx": float(np.sum(np.asarray(coeffs[0], dtype=np.float64) ** 2))
    }
    for i, detail_dict in enumerate(coeffs[1:], start=1):
        level_number = n_levels - i + 1
        total = sum(float(np.sum(np.asarray(v, dtype=np.float64) ** 2)) for v in detail_dict.values())
        energies[f"level_{level_number}_detail"] = total
    return energies


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------
def process_subject(subject_dir: Path, output_dir: Path, logger: logging.Logger) -> Optional[dict]:
    start_time = time.perf_counter()
    subject_id = subject_dir.name

    patches = np.load(subject_dir / "patches.npy")
    metadata_path = subject_dir / "metadata.csv"
    metadata_df = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()

    if patches.shape[0] == 0:
        logger.warning("%s: no patches to decompose, skipping.", subject_id)
        return None

    n_patches = patches.shape[0]

    coeff_arrays: dict[str, np.ndarray] = {}
    feature_arrays: dict[str, np.ndarray] = {}
    recon_mse_lists: dict[str, list[float]] = {name: [] for name in CHANNEL_NAMES}
    energy_ratio_lists: dict[str, list[float]] = {name: [] for name in CHANNEL_NAMES}
    level_energy_sums: dict[str, dict[str, float]] = {name: {} for name in CHANNEL_NAMES}
    sample_patch: dict[str, np.ndarray] = {}
    sample_coeff: dict[str, np.ndarray] = {}
    sample_recon: dict[str, np.ndarray] = {}

    for ch_idx, name in enumerate(CHANNEL_NAMES):
        coeff_list: list[np.ndarray] = []
        feature_list: list[list[float]] = []

        for p_idx in range(n_patches):
            patch = patches[p_idx, ch_idx]
            coeff_arr, coeff_slices, coeffs = decompose_patch(patch, WAVELET, DECOMP_LEVEL, PAD_MODE)
            coeff_list.append(coeff_arr.astype(np.float32))
            # float32 first, matching what Step 7 used to read back from disk
            feature_list.append(compute_token_features(coeff_list[-1]))

            recon = reconstruct_patch(coeff_arr, coeff_slices, WAVELET, PAD_MODE)
            recon = recon[: patch.shape[0], : patch.shape[1], : patch.shape[2]]
            mse = float(np.mean((patch.astype(np.float64) - recon) ** 2))
            recon_mse_lists[name].append(mse)

            energy_ratio_lists[name].append(compute_energy_ratio(patch, coeff_arr))

            for band, energy in per_level_energy(coeffs).items():
                level_energy_sums[name][band] = level_energy_sums[name].get(band, 0.0) + energy

            if p_idx == 0:
                sample_patch[name] = patch.copy()
                sample_coeff[name] = coeff_arr.copy()
                sample_recon[name] = recon.copy()

        coeff_arrays[name] = np.stack(coeff_list, axis=0)
        feature_arrays[name] = np.array(feature_list, dtype=np.float64)

    subject_out_dir = output_dir / subject_id
    subject_out_dir.mkdir(parents=True, exist_ok=True)
    for name in CHANNEL_NAMES:
        np.save(subject_out_dir / f"spectral_features_{name}.npy", feature_arrays[name])

    if not metadata_df.empty:
        metadata_out = metadata_df.copy()
        metadata_out["T1_Reconstruction_MSE"] = recon_mse_lists["T1"]
        metadata_out["T1_Energy_Ratio"] = energy_ratio_lists["T1"]
        metadata_out["FLAIR_Reconstruction_MSE"] = recon_mse_lists["FLAIR"]
        metadata_out["FLAIR_Energy_Ratio"] = energy_ratio_lists["FLAIR"]
        metadata_out.to_csv(subject_out_dir / "metadata.csv", index=False)

    validation: dict[str, dict] = {}
    for name in CHANNEL_NAMES:
        mse_arr = np.array(recon_mse_lists[name])
        ratio_arr = np.array(energy_ratio_lists[name])

        ratio_mean = float(np.mean(ratio_arr)) if ratio_arr.size else 0.0
        ratio_std = float(np.std(ratio_arr)) if ratio_arr.size else 0.0
        ratio_cv = (ratio_std / ratio_mean) if ratio_mean else 0.0

        # Reconstruction accuracy is the sole pass/fail gate (see CONFIG
        # note on ENERGY_RATIO_CV_TOLERANCE for why energy ratio isn't).
        passed = bool(mse_arr.size > 0 and np.max(mse_arr) <= RECON_MSE_THRESHOLD)

        validation[name] = {
            "Subject_ID": subject_id,
            "Channel": name,
            "N_Patches": int(mse_arr.size),
            "Mean_Reconstruction_MSE": float(np.mean(mse_arr)) if mse_arr.size else 0.0,
            "Max_Reconstruction_MSE": float(np.max(mse_arr)) if mse_arr.size else 0.0,
            "Mean_Energy_Ratio": ratio_mean,
            "Std_Energy_Ratio": ratio_std,
            "Energy_Ratio_CV": ratio_cv,
            "Energy_Consistent": bool(ratio_cv <= ENERGY_RATIO_CV_TOLERANCE),
            "Passed": passed,
        }

    elapsed = time.perf_counter() - start_time

    return {
        "subject_id": subject_id,
        "elapsed_seconds": elapsed,
        "n_patches": n_patches,
        "validation": validation,
        "subject_summary": {
            "Subject_ID": subject_id,
            "N_Patches": n_patches,
            "T1_Passed": validation["T1"]["Passed"],
            "FLAIR_Passed": validation["FLAIR"]["Passed"],
            "Output_Path": str(subject_out_dir),
        },
        "coeff_arrays": coeff_arrays,
        "energy_ratios": energy_ratio_lists,
        "level_energy_sum": level_energy_sums,
        "sample_patch": sample_patch,
        "sample_coeff": sample_coeff,
        "sample_recon": sample_recon,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_wavelet_coefficients(sample_coeff: dict[str, np.ndarray], figures_dir: Path, logger: logging.Logger) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, name in zip(axes, CHANNEL_NAMES):
        arr = sample_coeff[name]
        mid = arr.shape[2] // 2
        im = ax.imshow(np.log1p(np.abs(arr[:, :, mid])), cmap="viridis")
        ax.set_title(f"{name} wavelet coefficients (log|.|, mid slice)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures_dir / "wavelet_coefficient_visualization.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved wavelet_coefficient_visualization.png")


def plot_energy_distribution(all_energy_ratios: dict[str, list[float]], figures_dir: Path, logger: logging.Logger) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in zip(CHANNEL_NAMES, ("steelblue", "darkorange")):
        values = all_energy_ratios[name]
        if not values:
            continue
        ax.hist(values, bins=40, alpha=0.55, label=name, color=color, edgecolor="black")
    ax.axvline(1.0, color="red", linestyle="--", label="Perfect conservation (ratio=1)")
    ax.set_xlabel("Coefficient energy / original patch energy")
    ax.set_ylabel("Patch count")
    ax.set_title("Energy conservation across all patches")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "energy_distribution.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved energy_distribution.png")


def plot_reconstruction_comparison(
    sample_patch: dict[str, np.ndarray], sample_recon: dict[str, np.ndarray],
    figures_dir: Path, logger: logging.Logger,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, name in enumerate(CHANNEL_NAMES):
        original = sample_patch[name]
        recon = sample_recon[name]
        mid = original.shape[2] // 2
        diff = np.abs(original.astype(np.float64) - recon)

        axes[row, 0].imshow(original[:, :, mid], cmap="gray")
        axes[row, 0].set_title(f"{name} original")
        axes[row, 1].imshow(recon[:, :, mid], cmap="gray")
        axes[row, 1].set_title(f"{name} reconstructed")
        im = axes[row, 2].imshow(diff[:, :, mid], cmap="inferno")
        axes[row, 2].set_title(f"{name} |difference|")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046)
        for c in range(3):
            axes[row, c].axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "reconstruction_comparison.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved reconstruction_comparison.png")


def plot_power_spectrum(
    level_energy_totals: dict[str, dict[str, float]],
    figures_dir: Path, logger: logging.Logger,
) -> None:
    band_order = ["approx", "level_3_detail", "level_2_detail", "level_1_detail"]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = np.arange(len(band_order))

    for offset, name, color in ((-width / 2, "T1", "steelblue"), (width / 2, "FLAIR", "darkorange")):
        totals = level_energy_totals[name]
        grand_total = sum(totals.values())
        fractions = [totals.get(band, 0.0) / grand_total if grand_total else 0.0 for band in band_order]
        ax.bar(x + offset, fractions, width=width, label=name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(["Approx\n(coarsest)", "Level 3\ndetail", "Level 2\ndetail", "Level 1\ndetail\n(finest)"])
    ax.set_ylabel("Fraction of total coefficient energy")
    ax.set_title("Mean energy distribution across decomposition levels (power spectrum)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "power_spectrum.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved power_spectrum.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for d in (SPECTRAL_OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(LOGS_DIR / "step6.log")
    logger.info(
        "Starting Step 6: Localized Spectral Decomposition (wavelet=%s, level=%d, mode=%s)",
        WAVELET, DECOMP_LEVEL, PAD_MODE,
    )

    subject_dirs = discover_subjects(PATCH_INPUT_DIR)
    if not subject_dirs:
        logger.error("No subjects with patches.npy found in %s", PATCH_INPUT_DIR)
        return

    n_subjects = len(subject_dirs)
    validation_rows: list[dict] = []
    runtime_rows: list[dict] = []
    subject_summary_rows: list[dict] = []
    coeff_stats = {name: RunningStats() for name in CHANNEL_NAMES}
    level_energy_totals: dict[str, dict[str, float]] = {name: {} for name in CHANNEL_NAMES}
    total_patch_count = {name: 0 for name in CHANNEL_NAMES}
    all_energy_ratios: dict[str, list[float]] = {name: [] for name in CHANNEL_NAMES}
    vis_sample: Optional[dict] = None
    n_failed = 0

    pipeline_start = time.perf_counter()

    for i, subject_dir in enumerate(tqdm(subject_dirs, desc="Subjects"), start=1):
        logger.info("Processing subject %d / %d (%s)", i, n_subjects, subject_dir.name)
        try:
            result = process_subject(subject_dir, SPECTRAL_OUTPUT_DIR, logger)
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.error("Failed subject %s: %s", subject_dir.name, exc)
            n_failed += 1
            continue

        if result is None:
            continue

        runtime_rows.append({
            "subject_id": result["subject_id"],
            "elapsed_seconds": round(result["elapsed_seconds"], 3),
        })
        subject_summary_rows.append(result["subject_summary"])

        for name in CHANNEL_NAMES:
            validation_rows.append(result["validation"][name])
            coeff_stats[name].update(result["coeff_arrays"][name])
            total_patch_count[name] += result["n_patches"]
            all_energy_ratios[name].extend(result["energy_ratios"][name])
            for band, energy in result["level_energy_sum"][name].items():
                level_energy_totals[name][band] = level_energy_totals[name].get(band, 0.0) + energy

        if vis_sample is None:
            vis_sample = {
                "sample_patch": result["sample_patch"],
                "sample_coeff": result["sample_coeff"],
                "sample_recon": result["sample_recon"],
            }

        # result["coeff_arrays"] can now be garbage collected; only the
        # streamed stats/aggregates above are kept in memory going forward.

    total_runtime = time.perf_counter() - pipeline_start
    runtime_rows.append({"subject_id": "TOTAL", "elapsed_seconds": round(total_runtime, 3)})

    total_patches_processed = total_patch_count[CHANNEL_NAMES[0]]

    summary_rows = list(subject_summary_rows)
    summary_rows.append({
        "Subject_ID": "TOTAL",
        "N_Patches": total_patches_processed,
        "T1_Passed": sum(1 for r in subject_summary_rows if r["T1_Passed"]),
        "FLAIR_Passed": sum(1 for r in subject_summary_rows if r["FLAIR_Passed"]),
        "Output_Path": str(SPECTRAL_OUTPUT_DIR),
    })
    pd.DataFrame(summary_rows).to_csv(METRICS_DIR / "summary.csv", index=False)

    pd.DataFrame(validation_rows).to_csv(METRICS_DIR / "validation.csv", index=False)

    statistics_rows = []
    for name in CHANNEL_NAMES:
        stats_obj = coeff_stats[name]
        ratios = np.array(all_energy_ratios[name]) if all_energy_ratios[name] else np.array([])
        statistics_rows.append({
            "Channel": name,
            "Coefficient_Count": stats_obj.count,
            "Coefficient_Mean": stats_obj.mean(),
            "Coefficient_Std": stats_obj.std(),
            "Coefficient_Min": stats_obj.min_val if stats_obj.count else 0.0,
            "Coefficient_Max": stats_obj.max_val if stats_obj.count else 0.0,
            "Dataset_Mean_Energy_Ratio": float(np.mean(ratios)) if ratios.size else 0.0,
            "Dataset_Std_Energy_Ratio": float(np.std(ratios)) if ratios.size else 0.0,
        })
    pd.DataFrame(statistics_rows).to_csv(METRICS_DIR / "statistics.csv", index=False)

    pd.DataFrame(runtime_rows).to_csv(METRICS_DIR / "runtime.csv", index=False)

    if vis_sample is not None:
        plot_wavelet_coefficients(vis_sample["sample_coeff"], FIGURES_DIR, logger)
        plot_reconstruction_comparison(vis_sample["sample_patch"], vis_sample["sample_recon"], FIGURES_DIR, logger)
    else:
        logger.warning("No sample patch available for coefficient/reconstruction visualizations.")

    if any(all_energy_ratios[name] for name in CHANNEL_NAMES):
        plot_energy_distribution(all_energy_ratios, FIGURES_DIR, logger)

    if any(level_energy_totals[name] for name in CHANNEL_NAMES):
        plot_power_spectrum(level_energy_totals, FIGURES_DIR, logger)

    logger.info(
        "Step 6 complete. Subjects processed: %d/%d (failed: %d) | Total runtime: %.2f s",
        len(subject_summary_rows), n_subjects, n_failed, total_runtime,
    )


if __name__ == "__main__":
    main()
