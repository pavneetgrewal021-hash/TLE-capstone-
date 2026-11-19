from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from nibabel.processing import resample_to_output
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from pipeline_config import RUN_DIR, load_subject_list

BASE_DIR = Path(RUN_DIR)
NORMALIZED_DIR = BASE_DIR / "normalized_outputs"
PATCH_OUTPUT_DIR = BASE_DIR / "patch_outputs"
FIGURES_DIR = BASE_DIR / "figures" / "step5"
METRICS_DIR = BASE_DIR / "metrics" / "step5"
LOGS_DIR = BASE_DIR / "logs"
ATLAS_LABELS_JSON = BASE_DIR / "atlas_labels.json"

PATCH_SIZE = 32
STRIDE = 16
# Every volume is resampled to this isotropic voxel size before patching,
# so PATCH_SIZE voxels = PATCH_SIZE mm for every subject (see load_volume).
TARGET_VOXEL_MM = 1.0
N_CHANNELS = 2
BACKGROUND_EPS = 1e-6
BACKGROUND_MAX_FRACTION = 0.70
DPI = 300
RANDOM_SEED = 42

# Filename keyword matching (case-insensitive, applied to *.nii.gz in each
# subject folder). Adjust these if step1-4 outputs use different naming.
T1_KEYWORDS = ["t1"]
FLAIR_KEYWORDS = ["flair"]
ATLAS_KEYWORDS = ["atlas"]

METADATA_COLUMNS = [
    "Subject_ID",
    "Patch_ID",
    "Center_Coordinate",
    "Atlas_Region",
    "Hemisphere",
    "Background_Percentage",
]


@dataclass
class SubjectPaths:
    subject_id: str
    t1: Path
    flair: Path
    atlas: Optional[Path]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step5_patch_extraction")
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
# Discovery / IO helpers
# ---------------------------------------------------------------------------
def discover_subjects(normalized_dir: Path) -> list[Path]:
    if not normalized_dir.exists():
        return []
    selected = load_subject_list()
    return sorted(p for p in normalized_dir.iterdir() if p.is_dir() and p.name in selected)


def find_file(directory: Path, must_include: list[str]) -> Path:
    candidates = sorted(
        p for p in directory.rglob("*.nii.gz")
        if all(k.lower() in p.name.lower() for k in must_include)
    )
    if not candidates:
        raise FileNotFoundError(f"No file matching keywords {must_include} found in {directory}")
    return candidates[0]


def get_subject_paths(subject_dir: Path) -> SubjectPaths:
    subject_id = subject_dir.name
    t1 = find_file(subject_dir, T1_KEYWORDS)
    flair = find_file(subject_dir, FLAIR_KEYWORDS)
    try:
        atlas: Optional[Path] = find_file(subject_dir, ATLAS_KEYWORDS)
    except FileNotFoundError:
        atlas = None
    return SubjectPaths(subject_id=subject_id, t1=t1, flair=flair, atlas=atlas)


def load_atlas_label_map(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def load_volume(path: Path, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Loads a volume resampled to TARGET_VOXEL_MM isotropic (used for an
    atlas, which must already be in MNI space to line up with the
    MNI-aligned T1/FLAIR). order=1 (linear) for intensities, 0 for labels."""
    img = nib.load(str(path))
    img = resample_to_output(img, voxel_sizes=(TARGET_VOXEL_MM,) * 3, order=order, mode="constant", cval=0.0)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    return data, img.affine


_MNI_REFERENCE: dict = {}


def mni_reference(cache_dir: Path) -> tuple[sitk.Image, sitk.Image, np.ndarray]:
    """nilearn's bundled 1 mm MNI152 T1 template, brain-masked (the subject
    T1s are skull-stripped), plus its brain mask, as SimpleITK images."""
    if not _MNI_REFERENCE:
        template = load_mni152_template(resolution=1)
        mask = load_mni152_brain_mask(resolution=1)
        brain = nib.Nifti1Image(
            (template.get_fdata() * (mask.get_fdata() > 0)).astype(np.float32), template.affine
        )
        brain_path = cache_dir / "_mni152_brain_1mm.nii.gz"
        mask_path = cache_dir / "_mni152_brain_mask_1mm.nii.gz"
        cache_dir.mkdir(parents=True, exist_ok=True)
        nib.save(brain, str(brain_path))
        nib.save(nib.Nifti1Image((mask.get_fdata() > 0).astype(np.float32), template.affine), str(mask_path))
        _MNI_REFERENCE["brain"] = sitk.ReadImage(str(brain_path), sitk.sitkFloat32)
        _MNI_REFERENCE["mask"] = sitk.ReadImage(str(mask_path), sitk.sitkFloat32)
        _MNI_REFERENCE["affine"] = template.affine
    return _MNI_REFERENCE["brain"], _MNI_REFERENCE["mask"], _MNI_REFERENCE["affine"]


def align_to_mni(t1_path: Path, flair_path: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rigidly (6 DOF: rotation + translation, no scaling or warping)
    aligns the T1 to the MNI152 template and resamples T1 and FLAIR onto
    the template's 1 mm grid.

    Why: hemispheres are assigned by the sign of world x, which is only the
    true midline if the brain is centred and unrotated. In native scanner
    space it often isn't - sub-4001 had 629k brain voxels at x<0 vs 801k at
    x>0 - so patches were being put in the wrong hemisphere graph. After
    this alignment x=0 is the midsagittal plane for every subject, and every
    patch is exactly PATCH_SIZE mm (1 mm grid) regardless of acquisition
    resolution. Rigid-only keeps each brain's own shape and texture, which
    is what the wavelet features measure.

    Returns (t1, flair) as (x, y, z) arrays on the template grid, the
    template affine, and the final Mattes MI (more negative = better)."""
    ref_brain, ref_mask, ref_affine = mni_reference(cache_dir)
    t1 = sitk.ReadImage(str(t1_path), sitk.sitkFloat32)
    flair = sitk.ReadImage(str(flair_path), sitk.sitkFloat32)

    # Initialise by matching brain-mask centroids. (MOMENTS on the
    # WhiteStripe-normalized T1 itself would be skewed by its negative
    # grey-matter/CSF values.)
    moving_mask = sitk.Cast(sitk.NotEqual(t1, 0.0), sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        ref_mask, moving_mask, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1, RANDOM_SEED)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    transform = reg.Execute(ref_brain, t1)

    # sitk arrays are (z, y, x); transpose to nibabel's (x, y, z) so the
    # template affine applies directly.
    t1_mni = sitk.GetArrayFromImage(
        sitk.Resample(t1, ref_brain, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    ).transpose(2, 1, 0)
    flair_mni = sitk.GetArrayFromImage(
        sitk.Resample(flair, ref_brain, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    ).transpose(2, 1, 0)

    # The template registration fixes rotation but can leave the brain
    # shifted sideways (MI is tolerant to that when the brain is smaller
    # than the template: sub-4001 ended up centred ~4.5 mm left of x=0).
    # Snap the midline to x=0 using the brain's own left-right symmetry.
    mid_voxel = int(round(np.linalg.inv(ref_affine)[0] @ np.array([0.0, 0.0, 0.0, 1.0])))
    shift = mid_voxel - find_symmetry_plane(np.abs(t1_mni) > BACKGROUND_EPS)
    t1_mni = shift_along_x(t1_mni, shift)
    flair_mni = shift_along_x(flair_mni, shift)
    return t1_mni, flair_mni, ref_affine, reg.GetMetricValue()


def find_symmetry_plane(brain: np.ndarray, search_mm: int = 20) -> int:
    """x-index (on the 1 mm grid) of the sagittal plane about which the
    brain mask best matches its own mirror image - the midsagittal plane,
    to 1 mm. Searched within +/- search_mm of the brain's bounding-box
    centre."""
    columns = np.nonzero(brain.any(axis=(1, 2)))[0]
    centre = (columns.min() + columns.max()) // 2
    best_c, best_overlap = centre, -1
    for c in range(centre - search_mm, centre + search_mm + 1):
        lo = max(0, 2 * c - brain.shape[0] + 1)
        hi = min(brain.shape[0], 2 * c + 1)
        overlap = np.logical_and(brain[lo:hi], brain[lo:hi][::-1]).sum()
        if overlap > best_overlap:
            best_c, best_overlap = c, overlap
    return best_c


def shift_along_x(volume: np.ndarray, shift: int) -> np.ndarray:
    """Whole-voxel shift along x with zero fill (no interpolation, so the
    texture the wavelets measure is untouched)."""
    out = np.zeros_like(volume)
    if shift > 0:
        out[shift:] = volume[:-shift]
    elif shift < 0:
        out[:shift] = volume[-shift:]
    else:
        out[:] = volume
    return out


def validate_image_alignment(t1_path: Path, flair_path: Path, logger: logging.Logger) -> None:
    t1_img = sitk.ReadImage(str(t1_path))
    flair_img = sitk.ReadImage(str(flair_path))
    if t1_img.GetSize() != flair_img.GetSize():
        logger.warning(
            "Size mismatch for %s: T1 %s vs FLAIR %s",
            t1_path.parent.name, t1_img.GetSize(), flair_img.GetSize(),
        )
    spacing_diff = max(abs(a - b) for a, b in zip(t1_img.GetSpacing(), flair_img.GetSpacing()))
    if spacing_diff > 1e-3:
        logger.warning("Spacing mismatch (%.4f mm) for %s", spacing_diff, t1_path.parent.name)


def compute_hemisphere(center: tuple[int, int, int], affine: np.ndarray) -> str:
    # Assumes RAS+ world orientation (standard NIfTI convention): +X = Right.
    world = affine @ np.array([center[0], center[1], center[2], 1.0])
    if world[0] > 0:
        return "R"
    if world[0] < 0:
        return "L"
    return "Midline"


def crosses_midline(sx: int, center: tuple[int, int, int], affine: np.ndarray) -> bool:
    """True if the patch's x-extent spans world x=0. Such a patch holds
    tissue from BOTH hemispheres, so assigning it to one hemisphere graph
    (by its center) would blur exactly the L/R asymmetry Steps 9-11 rely
    on. Checked at the patch's first and last voxel along x."""
    first = affine @ np.array([sx, center[1], center[2], 1.0])
    last = affine @ np.array([sx + PATCH_SIZE - 1, center[1], center[2], 1.0])
    # strict: a patch whose edge voxel sits exactly ON x=0 does not cross
    # (with the midline-anchored grid, the patches either side start/end there)
    lo, hi = sorted((first[0], last[0]))
    return lo < 0 < hi


def generate_grid_starts(dim: int, patch_size: int, stride: int, offset: int = 0) -> list[int]:
    if dim < patch_size:
        return []
    return list(range(offset, dim - patch_size + 1, stride))


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def process_subject(
    subject_dir: Path,
    atlas_label_map: dict[int, str],
    patch_output_dir: Path,
    logger: logging.Logger,
) -> dict:
    start_time = time.perf_counter()
    paths = get_subject_paths(subject_dir)
    validate_image_alignment(paths.t1, paths.flair, logger)

    t1_data, flair_data, affine, mni_mi = align_to_mni(paths.t1, paths.flair, patch_output_dir)

    if t1_data.shape != flair_data.shape:
        raise ValueError(
            f"T1/FLAIR shape mismatch for {paths.subject_id}: {t1_data.shape} vs {flair_data.shape}"
        )

    atlas_data: Optional[np.ndarray] = None
    if paths.atlas is not None:
        atlas_data, _ = load_volume(paths.atlas, order=0)
        if atlas_data.shape != t1_data.shape:
            logger.warning("Atlas shape mismatch for %s. Ignoring atlas for this subject.", paths.subject_id)
            atlas_data = None

    # Anchor the x-grid on the midline (world x=0) instead of the image
    # edge, so both hemispheres are tiled identically (with an edge-anchored
    # grid, where the midline happened to fall between grid lines changed
    # the L/R patch counts). The MNI template grid is axis-aligned RAS, so
    # axis 0 is left->right.
    mid_x = int(round(np.linalg.inv(affine)[0] @ np.array([0.0, 0.0, 0.0, 1.0])))
    x_starts = generate_grid_starts(t1_data.shape[0], PATCH_SIZE, STRIDE, offset=mid_x % STRIDE)
    y_starts = generate_grid_starts(t1_data.shape[1], PATCH_SIZE, STRIDE)
    z_starts = generate_grid_starts(t1_data.shape[2], PATCH_SIZE, STRIDE)
    half = PATCH_SIZE // 2

    patches: list[np.ndarray] = []
    records: list[dict] = []
    background_all: list[float] = []
    patch_id = 0
    total_candidates = 0
    rejected_midline = 0

    for sx in x_starts:
        for sy in y_starts:
            for sz in z_starts:
                total_candidates += 1
                t1_patch = t1_data[sx:sx + PATCH_SIZE, sy:sy + PATCH_SIZE, sz:sz + PATCH_SIZE]
                flair_patch = flair_data[sx:sx + PATCH_SIZE, sy:sy + PATCH_SIZE, sz:sz + PATCH_SIZE]

                # Background is judged on BOTH channels: FLAIR has smaller
                # true coverage than T1 (see Steps 1/4), so a patch can be
                # full of T1 brain yet mostly out-of-FOV zeros in FLAIR.
                t1_bg = float(np.sum(np.abs(t1_patch) < BACKGROUND_EPS)) / float(t1_patch.size)
                flair_bg = float(np.sum(np.abs(flair_patch) < BACKGROUND_EPS)) / float(flair_patch.size)
                background_pct = max(t1_bg, flair_bg)
                background_all.append(background_pct)

                if background_pct > BACKGROUND_MAX_FRACTION:
                    continue

                center = (sx + half, sy + half, sz + half)
                if crosses_midline(sx, center, affine):
                    rejected_midline += 1
                    continue

                if atlas_data is not None:
                    label_value = int(atlas_data[center])
                    if label_value == 0:
                        atlas_region = "Background"
                    else:
                        atlas_region = atlas_label_map.get(label_value, f"label_{label_value}")
                else:
                    atlas_region = "NA"

                hemisphere = compute_hemisphere(center, affine)
                patch = np.stack([t1_patch, flair_patch], axis=0).astype(np.float32)
                patches.append(patch)

                records.append({
                    "Subject_ID": paths.subject_id,
                    "Patch_ID": patch_id,
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "Center_Coordinate": f"({center[0]}, {center[1]}, {center[2]})",
                    "Atlas_Region": atlas_region,
                    "Hemisphere": hemisphere,
                    "Background_Percentage": round(background_pct * 100.0, 3),
                })
                patch_id += 1

    patch_array = (
        np.stack(patches, axis=0) if patches
        else np.empty((0, N_CHANNELS, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    )

    subject_out_dir = patch_output_dir / paths.subject_id
    subject_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(subject_out_dir / "patches.npy", patch_array)

    metadata_df = pd.DataFrame(records)
    if metadata_df.empty:
        metadata_df = pd.DataFrame(columns=METADATA_COLUMNS)
    else:
        metadata_df = metadata_df[METADATA_COLUMNS]
    metadata_df.to_csv(subject_out_dir / "metadata.csv", index=False)

    present_regions: set[str] = set()
    if atlas_data is not None:
        for lv in np.unique(atlas_data):
            lv_int = int(lv)
            if lv_int == 0:
                continue
            present_regions.add(atlas_label_map.get(lv_int, f"label_{lv_int}"))
    captured_regions = {r["Atlas_Region"] for r in records if r["Atlas_Region"] not in ("NA", "Background")}

    left_count = sum(1 for r in records if r["Hemisphere"] == "L")
    right_count = sum(1 for r in records if r["Hemisphere"] == "R")
    elapsed = time.perf_counter() - start_time

    subject_row = {
        "subject_id": paths.subject_id,
        "total_candidates": total_candidates,
        "accepted": len(records),
        "rejected": total_candidates - len(records),
        "rejected_midline": rejected_midline,
        "mni_rigid_mi": round(float(mni_mi), 4),
        "coverage": round(len(records) / total_candidates, 4) if total_candidates else 0.0,
        "left_hemisphere_patches": left_count,
        "right_hemisphere_patches": right_count,
        "atlas_available": atlas_data is not None,
    }

    return {
        "subject_id": paths.subject_id,
        "subject_row": subject_row,
        "elapsed_seconds": elapsed,
        "records": records,
        "background_all": background_all,
        "present_regions": present_regions,
        "captured_regions": captured_regions,
        "patch_array": patch_array,
        "t1_data": t1_data,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_patch_overlay(t1_data: np.ndarray, records: list[dict], figures_dir: Path, logger: logging.Logger) -> None:
    slice_idx = t1_data.shape[2] // 2
    half = PATCH_SIZE // 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(t1_data[:, :, slice_idx].T, cmap="gray", origin="lower")
    for rec in records:
        if abs(rec["center_z"] - slice_idx) > half:
            continue
        sx = rec["center_x"] - half
        sy = rec["center_y"] - half
        rect = plt.Rectangle((sx, sy), PATCH_SIZE, PATCH_SIZE, edgecolor="red", facecolor="none", linewidth=0.6)
        ax.add_patch(rect)
    ax.set_title(f"Accepted patch overlay — axial slice {slice_idx}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "patch_overlay.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved patch_overlay.png")


def plot_random_patches(patch_array: np.ndarray, figures_dir: Path, logger: logging.Logger, n: int = 9) -> None:
    if patch_array.shape[0] == 0:
        logger.warning("No accepted patches available for random-patch visualization.")
        return

    rng = np.random.default_rng(RANDOM_SEED)
    n = min(n, patch_array.shape[0])
    indices = rng.choice(patch_array.shape[0], size=n, replace=False)
    mid = PATCH_SIZE // 2
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.2))
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        axes[r, c * 2].imshow(patch_array[idx, 0, :, :, mid], cmap="gray")
        axes[r, c * 2].set_title(f"T1 #{idx}", fontsize=8)
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].imshow(patch_array[idx, 1, :, :, mid], cmap="gray")
        axes[r, c * 2 + 1].set_title(f"FLAIR #{idx}", fontsize=8)
        axes[r, c * 2 + 1].axis("off")

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].axis("off")

    fig.tight_layout()
    fig.savefig(figures_dir / "random_patches.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved random_patches.png")


def plot_background_histogram(background_all: list[float], figures_dir: Path, logger: logging.Logger) -> None:
    values = np.array(background_all) * 100.0
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=40, color="steelblue", edgecolor="black")
    ax.axvline(BACKGROUND_MAX_FRACTION * 100.0, color="red", linestyle="--", label="Rejection threshold")
    ax.set_xlabel("Background percentage (%)")
    ax.set_ylabel("Candidate patch count")
    ax.set_title("Background percentage distribution (all candidate patches)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "background_histogram.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved background_histogram.png")


def plot_atlas_region_distribution(records_df: pd.DataFrame, figures_dir: Path, logger: logging.Logger) -> None:
    counts = records_df["Atlas_Region"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.3), 5))
    ax.bar(counts.index.astype(str), counts.values, color="darkseagreen")
    ax.set_xticklabels(counts.index.astype(str), rotation=90, fontsize=6)
    ax.set_ylabel("Accepted patch count")
    ax.set_title("Atlas region patch distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "atlas_region_distribution.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved atlas_region_distribution.png")


def plot_hemisphere_distribution(records_df: pd.DataFrame, figures_dir: Path, logger: logging.Logger) -> None:
    counts = records_df["Hemisphere"].value_counts()
    palette = ["#2c7fb8", "#de6b35", "#999999"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(counts.index.astype(str), counts.values, color=palette[: len(counts)])
    ax.set_ylabel("Accepted patch count")
    ax.set_title("Left vs Right hemisphere patch count")
    fig.tight_layout()
    fig.savefig(figures_dir / "hemisphere_distribution.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved hemisphere_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for d in (PATCH_OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(LOGS_DIR / "step5.log")
    logger.info("Starting Step 5: Patch Extraction")

    atlas_label_map = load_atlas_label_map(ATLAS_LABELS_JSON)
    subject_dirs = discover_subjects(NORMALIZED_DIR)

    if not subject_dirs:
        logger.error("No subjects found in %s", NORMALIZED_DIR)
        return

    n_subjects = len(subject_dirs)
    subject_rows: list[dict] = []
    runtime_rows: list[dict] = []
    all_records: list[dict] = []
    all_background: list[float] = []
    present_regions_all: set[str] = set()
    captured_regions_all: set[str] = set()
    vis_sample: Optional[dict] = None

    pipeline_start = time.perf_counter()

    for i, subject_dir in enumerate(tqdm(subject_dirs, desc="Subjects"), start=1):
        logger.info("Processing subject %d / %d (%s)", i, n_subjects, subject_dir.name)
        try:
            result = process_subject(subject_dir, atlas_label_map, PATCH_OUTPUT_DIR, logger)
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.error("Failed subject %s: %s", subject_dir.name, exc)
            continue

        subject_rows.append(result["subject_row"])
        runtime_rows.append({"subject_id": result["subject_id"], "elapsed_seconds": round(result["elapsed_seconds"], 3)})
        all_records.extend(result["records"])
        all_background.extend(result["background_all"])
        present_regions_all |= result["present_regions"]
        captured_regions_all |= result["captured_regions"]

        if vis_sample is None and result["patch_array"].shape[0] > 0:
            vis_sample = {
                "t1_data": result["t1_data"],
                "records": result["records"],
                "patch_array": result["patch_array"],
            }

    total_runtime = time.perf_counter() - pipeline_start
    runtime_rows.append({"subject_id": "TOTAL", "elapsed_seconds": round(total_runtime, 3)})

    records_df = pd.DataFrame(all_records)
    subject_df = pd.DataFrame(subject_rows)
    runtime_df = pd.DataFrame(runtime_rows)

    total_patches = int(subject_df["total_candidates"].sum()) if not subject_df.empty else 0
    accepted_patches = int(subject_df["accepted"].sum()) if not subject_df.empty else 0
    rejected_patches = total_patches - accepted_patches
    coverage = accepted_patches / total_patches if total_patches else 0.0

    pd.DataFrame([{
        "total_patches": total_patches,
        "accepted_patches": accepted_patches,
        "rejected_patches": rejected_patches,
        "patch_coverage": round(coverage, 4),
        "n_subjects_found": n_subjects,
        "n_subjects_processed": len(subject_rows),
    }]).to_csv(METRICS_DIR / "summary.csv", index=False)

    missing_regions = sorted(present_regions_all - captured_regions_all)
    left_count = int((records_df["Hemisphere"] == "L").sum()) if not records_df.empty else 0
    right_count = int((records_df["Hemisphere"] == "R").sum()) if not records_df.empty else 0
    region_coverage = (len(captured_regions_all) / len(present_regions_all)) if present_regions_all else 0.0

    pd.DataFrame([{
        "present_atlas_regions": len(present_regions_all),
        "captured_atlas_regions": len(captured_regions_all),
        "atlas_region_coverage": round(region_coverage, 4),
        "missing_atlas_regions": ";".join(missing_regions) if missing_regions else "None",
        "left_hemisphere_patches": left_count,
        "right_hemisphere_patches": right_count,
    }]).to_csv(METRICS_DIR / "validation.csv", index=False)

    bg_array = np.array(all_background) * 100.0 if all_background else np.array([])
    accepted_bg = records_df["Background_Percentage"].to_numpy() if not records_df.empty else np.array([])
    patches_per_subject = subject_df["accepted"].to_numpy() if not subject_df.empty else np.array([])

    if bg_array.size > 1:
        desc = scipy_stats.describe(bg_array)
        bg_skew, bg_kurt = float(desc.skewness), float(desc.kurtosis)
    else:
        bg_skew, bg_kurt = 0.0, 0.0

    pd.DataFrame([{
        "background_pct_mean_all": float(np.mean(bg_array)) if bg_array.size else 0.0,
        "background_pct_std_all": float(np.std(bg_array)) if bg_array.size else 0.0,
        "background_pct_min_all": float(np.min(bg_array)) if bg_array.size else 0.0,
        "background_pct_max_all": float(np.max(bg_array)) if bg_array.size else 0.0,
        "background_pct_skewness_all": bg_skew,
        "background_pct_kurtosis_all": bg_kurt,
        "background_pct_mean_accepted": float(np.mean(accepted_bg)) if accepted_bg.size else 0.0,
        "background_pct_std_accepted": float(np.std(accepted_bg)) if accepted_bg.size else 0.0,
        "patches_per_subject_mean": float(np.mean(patches_per_subject)) if patches_per_subject.size else 0.0,
        "patches_per_subject_std": float(np.std(patches_per_subject)) if patches_per_subject.size else 0.0,
    }]).to_csv(METRICS_DIR / "statistics.csv", index=False)

    subject_df.to_csv(METRICS_DIR / "subject_metrics.csv", index=False)
    runtime_df.to_csv(METRICS_DIR / "runtime.csv", index=False)

    if vis_sample is not None:
        plot_patch_overlay(vis_sample["t1_data"], vis_sample["records"], FIGURES_DIR, logger)
        plot_random_patches(vis_sample["patch_array"], FIGURES_DIR, logger)
    else:
        logger.warning("No accepted patches available for slice/random-patch visualization.")

    if all_background:
        plot_background_histogram(all_background, FIGURES_DIR, logger)
    if not records_df.empty:
        plot_atlas_region_distribution(records_df, FIGURES_DIR, logger)
        plot_hemisphere_distribution(records_df, FIGURES_DIR, logger)

    logger.info(
        "Step 5 complete. Subjects processed: %d/%d | Accepted patches: %d | Total runtime: %.2f s",
        len(subject_rows), n_subjects, accepted_patches, total_runtime,
    )


if __name__ == "__main__":
    main()
