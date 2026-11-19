# ============================================
# STEP 4: INTENSITY NORMALIZATION
# ============================================
# Standalone script. Reads the BIAS-CORRECTED outputs from Step 3
# (biascorrected_outputs/{sub_id}/{sub_id}_T1w_biascorrected.nii.gz,
#  {sub_id}_FLAIR_biascorrected.nii.gz, {sub_id}_brain_mask.nii.gz).
#
# Install once (same as Steps 1-3):
#   pip install SimpleITK nibabel nilearn scipy numpy matplotlib pandas tqdm
#
# METHOD: WhiteStripe normalization (Shinohara et al. 2014) - finds the
# dominant intensity peak (white matter) within the brain mask, takes a
# band of voxels around that peak, and normalizes using that band's
# mean/std instead of the whole mixed-tissue mask's mean/std. More
# robust to inter-subject anatomical variation (GM/WM/CSF proportions)
# than plain z-score.
#
# FIXES APPLIED AFTER FIRST RUN (both real bugs, not cosmetic):
#
# 1. OUT-OF-FOV ZERO CONTAMINATION: FLAIR was registered into T1 space in
#    Step 1. Since this dataset's FLAIR is acquired at an oblique angle
#    with smaller field-of-view than T1 (discovered back in Step 1), some
#    T1-brain-mask voxels fall OUTSIDE where FLAIR actually has real
#    signal - those get filled with 0 by the resampling. The brain mask
#    (built from T1) doesn't know about this, so it happily includes
#    these fake zeros as "brain tissue" for FLAIR. For subjects with a
#    lot of this padding, the peak-finder was locking onto the artificial
#    zero-spike instead of real tissue. FIX: exclude near-zero voxels
#    from the "valid" set used for peak-finding, stripe stats, AND the
#    final normalized output (they stay 0, correctly representing "no
#    data here" rather than getting transformed into a huge z-score).
#
# 2. OVERLY NARROW STRIPE BAND: a 10%-width band around the peak was
#    often SO narrow that its std was tiny, which meant normalizing the
#    rest of the (much wider) real tissue distribution against it
#    produced enormous, meaningless z-scores. FIX: widened the band to
#    25%, and switched the spread measure from raw std (sensitive to a
#    few odd voxels in a small sample) to a robust estimate (MAD-based),
#    which is standard practice specifically to avoid this failure mode.

import os
import glob
import logging
from itertools import combinations

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import gaussian_kde
from scipy.ndimage import binary_erosion
from scipy.signal import find_peaks
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================

NORMALIZATION_METHOD = "whitestripe"  # "whitestripe" or "zscore"

CONFIG = {
    # WIDENED from 0.10 - see fix #2 above.
    "whitestripe_width_pct": 0.25,

    # Voxels at or below this raw intensity are treated as "no data"
    # (out-of-FOV padding, not real tissue) and excluded from all
    # statistics and from the normalized output.
    "valid_voxel_epsilon": 1e-3,

    # After excluding pure zero-padding, the boundary between real FLAIR
    # signal and padding is not a hard edge - resampling interpolation
    # BLENDS them, creating a thin shell of distorted "partial" values
    # (nonzero, but not real tissue either). Eroding the valid region
    # inward by this many voxels strips out most of that blended shell
    # before statistics are computed. This is an approximation - the
    # fully correct fix is an explicit FOV-tracking mask generated back
    # in Step 1 (resampling a constant-1 field alongside the intensities),
    # which could be added later if this isn't sufficient.
    "boundary_erosion_voxels": 2,

    "kde_subsample_n": 20000,

    # T1 only: a density mode counts as a tissue peak if its prominence is
    # at least this fraction of the tallest mode - high enough to ignore
    # KDE ripples in the bright tail, low enough to keep the WM peak even
    # when grey matter dominates the histogram.
    "t1_peak_min_prominence": 0.10,

    # A subject's peak is flagged if it's this many robust-MADs away
    # from the group median peak - catches subjects like the outlier
    # this dataset produced (a peak ~100x everyone else's).
    "peak_outlier_mad_threshold": 5.0,

    # Extreme z-scores after normalization (beyond this) get flagged as
    # a diagnostic - shouldn't happen often with the widened/robust band,
    # but worth surfacing if it does.
    "extreme_zscore_threshold": 10.0,
    "extreme_zscore_flag_fraction": 0.01,  # flag if >1% of voxels exceed it

    "ks_subsample_n": 3000,
    "ks_pvalue_flag_threshold": 0.05,
    "ks_statistic_flag_threshold": 0.15,
}

MAX_SUBJECTS = None

# If a subject's expected output files already exist, skip recomputing
# them. Set True to force reprocessing everyone.
OVERWRITE = False

from pipeline_config import RUN_DIR, load_subject_list, report_file

BIASCORRECTED_DIR = os.path.join(RUN_DIR, "biascorrected_outputs")
OUT_DIR = os.path.join(RUN_DIR, "normalized_outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

_RNG = np.random.default_rng(42)


def setup_logging(out_dir):
    logger = logging.getLogger("step4_intensity_normalization")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(os.path.join(out_dir, "step4_intensity_normalization.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


logger = logging.getLogger("step4_intensity_normalization")


# ============================================
# DISCOVER SUBJECTS
# ============================================

def find_biascorrected_subjects(biascorrected_dir):
    subject_dirs = sorted(
        d for d in glob.glob(os.path.join(biascorrected_dir, "sub-*")) if os.path.isdir(d)
    )
    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject folders found under BIASCORRECTED_DIR='{biascorrected_dir}'. "
            f"Make sure Step 3 has been run first."
        )

    subjects = {}
    for sub_dir in subject_dirs:
        sub_id = os.path.basename(sub_dir)
        t1_path = os.path.join(sub_dir, f"{sub_id}_T1w_biascorrected.nii.gz")
        flair_path = os.path.join(sub_dir, f"{sub_id}_FLAIR_biascorrected.nii.gz")
        mask_path = os.path.join(sub_dir, f"{sub_id}_brain_mask.nii.gz")

        if all(os.path.exists(p) for p in [t1_path, flair_path, mask_path]):
            subjects[sub_id] = {"t1": t1_path, "flair": flair_path, "mask": mask_path}
        else:
            msg = f"{sub_id}: missing expected files in {sub_dir}, skipping."
            print(msg)
            logger.warning(msg)

    return subjects


def normalization_outputs_exist(sub_id):
    sub_out_dir = os.path.join(OUT_DIR, sub_id)
    return all(
        os.path.exists(os.path.join(sub_out_dir, f"{sub_id}{suffix}"))
        for suffix in ["_T1w_normalized.nii.gz", "_FLAIR_normalized.nii.gz", "_brain_mask.nii.gz"]
    )


def load_previous_report(report_path):
    if os.path.exists(report_path):
        try:
            return pd.read_csv(report_path)
        except Exception as e:
            logger.warning(f"Could not read previous report at {report_path}: {e}")
    return pd.DataFrame()


def reload_valid_normalized_values(norm_path, mask_path):
    """For RESUMED subjects only (already normalized in a previous run):
    reconstructs the valid-voxel value array from the saved outputs
    instead of re-running whitestripe/z-score normalization, so group-
    level KS tests / histograms still include every subject without
    redoing the peak-finding step. Approximation: a genuinely-normalized
    value of exactly 0.0 would be excluded here, same as a padding
    voxel - negligible at the population sizes used for these group
    comparisons."""
    img = sitk.ReadImage(norm_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    valid = mask_arr & (arr != 0)
    return arr[valid].astype(np.float32)


# ============================================
# CORE NORMALIZATION
# ============================================

def find_intensity_peak(values, modality):
    """White-matter reference peak, following Shinohara et al. (2014):
    - T1: the RIGHT-MOST (brightest) prominent mode. White matter is the
      brightest tissue in T1, but it is not always the TALLEST mode - on
      the DISCOVERY MR750 T1s grey matter forms the taller peak. Taking
      the tallest peak (the previous behaviour) referenced ~half the
      subjects to grey matter and half to white matter, split by scanner:
      the chosen peak sat at ~0.6 of the 99th-percentile intensity on MR750
      vs ~0.8-0.9 on HDx, and 71.5% of subject pairs still had different
      normalized T1 distributions (KS test).
    - FLAIR: the tallest mode, which is normal-appearing white matter."""
    n = min(len(values), CONFIG["kde_subsample_n"])
    sample = _RNG.choice(values, size=n, replace=False)
    kde = gaussian_kde(sample)
    xs = np.linspace(sample.min(), sample.max(), 512)
    density = kde(xs)
    if modality != "T1":
        return xs[np.argmax(density)]

    peaks, _ = find_peaks(density, prominence=CONFIG["t1_peak_min_prominence"] * density.max())
    if len(peaks) == 0:
        return xs[np.argmax(density)]
    return xs[peaks[-1]]


def robust_spread(values):
    """MAD-based robust std estimate - less sensitive than raw std to a
    handful of odd voxels in a possibly small sample (the stripe band)."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return mad * 1.4826  # scale factor makes this consistent with std for normal data


def whitestripe_normalize(arr, valid_mask, modality):
    brain_values = arr[valid_mask]
    peak = find_intensity_peak(brain_values, modality)

    half_width = CONFIG["whitestripe_width_pct"] * peak
    lower, upper = peak - half_width, peak + half_width
    stripe_values = brain_values[(brain_values >= lower) & (brain_values <= upper)]

    if len(stripe_values) < 50:
        raise RuntimeError(
            f"WhiteStripe band contained only {len(stripe_values)} voxels - too few. "
            f"Peak={peak:.2f}, band=[{lower:.2f}, {upper:.2f}]"
        )

    stripe_mean = stripe_values.mean()
    stripe_spread = robust_spread(stripe_values)

    norm_arr = np.zeros_like(arr)
    norm_arr[valid_mask] = (brain_values - stripe_mean) / (stripe_spread + 1e-8)
    return norm_arr, peak, stripe_mean, stripe_spread


def zscore_normalize(arr, valid_mask):
    brain_values = arr[valid_mask]
    mean = brain_values.mean()
    std = brain_values.std()
    norm_arr = np.zeros_like(arr)
    norm_arr[valid_mask] = (brain_values - mean) / (std + 1e-8)
    return norm_arr, mean, std


def normalize_image(image_path, mask_img, method, modality):
    """Takes an already-loaded mask image (read once per subject in the
    driver, reused for T1 and FLAIR) instead of re-reading the mask file
    from disk on every call."""
    img = sitk.ReadImage(image_path, sitk.sitkFloat32)

    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    mask_arr = sitk.GetArrayFromImage(mask_img).astype(bool)

    # FIX #1: exclude near-zero out-of-FOV padding from "valid" voxels
    valid_mask = mask_arr & (arr > CONFIG["valid_voxel_epsilon"])
    pre_erosion_pct_excluded = 100 * (mask_arr.sum() - valid_mask.sum()) / mask_arr.sum() if mask_arr.sum() else 0

    # Only erode if real padding was detected (e.g. FLAIR, resampled in
    # Step 1). T1 is native/non-resampled with no padding, so eroding it
    # too would just discard good edge tissue for no reason.
    erosion_voxels = CONFIG["boundary_erosion_voxels"]
    if erosion_voxels > 0 and pre_erosion_pct_excluded > 1.0:
        valid_mask = binary_erosion(valid_mask, iterations=erosion_voxels)

    n_excluded = mask_arr.sum() - valid_mask.sum()
    pct_excluded = 100 * n_excluded / mask_arr.sum() if mask_arr.sum() else 0

    if method == "whitestripe":
        norm_arr, peak, ref_mean, ref_spread = whitestripe_normalize(arr, valid_mask, modality)
        extra_info = {"peak": peak, "reference_mean": ref_mean, "reference_std": ref_spread,
                      "pct_excluded_as_padding": pct_excluded}
    elif method == "zscore":
        norm_arr, ref_mean, ref_spread = zscore_normalize(arr, valid_mask)
        extra_info = {"peak": None, "reference_mean": ref_mean, "reference_std": ref_spread,
                      "pct_excluded_as_padding": pct_excluded}
    else:
        raise ValueError(f"Unknown NORMALIZATION_METHOD: {method}")

    norm_img = sitk.GetImageFromArray(norm_arr.astype(np.float32))
    norm_img.CopyInformation(img)

    return norm_img, extra_info, norm_arr[valid_mask]


# ============================================
# PER-SUBJECT VALIDATION
# ============================================

def validate_normalization(norm_values, extra_info, sub_id, modality, method):
    pct_excluded = extra_info["pct_excluded_as_padding"]
    fov_note = f", excluded {pct_excluded:.1f}% as out-of-FOV padding" if pct_excluded > 0.5 else ""

    extreme_frac = np.mean(np.abs(norm_values) > CONFIG["extreme_zscore_threshold"]) if len(norm_values) else 0

    # IMPORTANT: extreme z-score voxels in FLAIR are NOT necessarily an
    # artifact for this study. FLAIR is specifically used to detect
    # abnormally bright pathological tissue (e.g. hippocampal sclerosis
    # in TLE) - such tissue is SUPPOSED to stand out as an outlier
    # relative to normal-appearing white matter after normalization.
    # Gating subjects out based on "too many extreme voxels" risks
    # silently discarding exactly the patients with visible abnormalities,
    # which would be a serious problem in an epilepsy-focused dataset.
    # So: report this number for every subject (useful signal either way),
    # but do NOT use it to block saving. Only a basic sanity check (did
    # the math run without producing NaN/inf) gates saving now.
    has_nan_or_inf = not np.all(np.isfinite(norm_values)) if len(norm_values) else True
    passed = not has_nan_or_inf and len(norm_values) > 0

    status = "PASS" if passed else "FAIL"
    if method == "whitestripe":
        print(f"[{status}] Normalization ({modality}, whitestripe) - {sub_id} | "
              f"peak={extra_info['peak']:.2f}, stripe_mean={extra_info['reference_mean']:.2f}, "
              f"stripe_spread={extra_info['reference_std']:.2f}{fov_note} | "
              f"extreme-z voxels={100*extreme_frac:.2f}% (reported only, not a rejection criterion)")
    else:
        print(f"[{status}] Normalization ({modality}, zscore) - {sub_id} | "
              f"mean={extra_info['reference_mean']:.2f}, std={extra_info['reference_std']:.2f}"
              f"{fov_note} | extreme-z voxels={100*extreme_frac:.2f}% (reported only, not a rejection criterion)")

    return {
        "sub_id": sub_id, "modality": modality, "method": method,
        "post_mean": norm_values.mean() if len(norm_values) else np.nan,
        "post_std": norm_values.std() if len(norm_values) else np.nan,
        "peak": extra_info.get("peak"),
        "pct_excluded_as_padding": pct_excluded,
        "extreme_zscore_pct": 100 * extreme_frac,
        "passed": passed,
    }


def flag_peak_outliers(report_df, modality):
    """Flags subjects whose peak is way off from the group - catches
    upstream anomalies (e.g. a subject that went through Step 1-3
    differently than everyone else) rather than a normalization bug."""
    subset = report_df[(report_df["modality"] == modality) & report_df["peak"].notna()]
    if len(subset) < 3:
        return
    peaks = subset["peak"].values
    median = np.median(peaks)
    mad = np.median(np.abs(peaks - median)) * 1.4826
    if mad == 0:
        return
    for _, row in subset.iterrows():
        z = abs(row["peak"] - median) / mad
        if z > CONFIG["peak_outlier_mad_threshold"]:
            print(f"  ANOMALY FLAG: {row['sub_id']} {modality} peak={row['peak']:.2f} is "
                  f"{z:.1f} robust-MADs from group median ({median:.2f}) - "
                  f"worth checking this subject's Steps 1-3 outputs individually")


# ============================================
# GROUP-LEVEL VALIDATION
# ============================================

def group_level_ks_test(norm_values_by_subject, modality):
    sub_ids = list(norm_values_by_subject.keys())
    n_sub = CONFIG["ks_subsample_n"]

    subsampled = {
        sid: _RNG.choice(vals, size=min(len(vals), n_sub), replace=False)
        for sid, vals in norm_values_by_subject.items() if len(vals) > 0
    }

    results = []
    for a, b in combinations(subsampled.keys(), 2):
        stat, p = stats.ks_2samp(subsampled[a], subsampled[b])
        flagged = (p < CONFIG["ks_pvalue_flag_threshold"]) and (stat > CONFIG["ks_statistic_flag_threshold"])
        results.append({
            "modality": modality, "pair": f"{a} vs {b}",
            "ks_stat": stat, "p_value": p, "flagged_different": flagged,
        })
    return pd.DataFrame(results)


def save_histogram_comparison(norm_values_by_subject, modality, method):
    plt.figure(figsize=(8, 5))
    shared_bins = np.linspace(-5, 8, 90)
    for sub_id, values in norm_values_by_subject.items():
        if len(values) == 0:
            continue
        plt.hist(values, bins=shared_bins, histtype="step", density=True, alpha=0.6, label=sub_id)
    plt.title(f"{modality} ({method}): normalized brain-tissue intensity distributions")
    plt.xlabel("normalized intensity")
    plt.ylabel("density")
    plt.legend(fontsize=6, ncol=2)
    out_path = os.path.join(FIG_DIR, f"group_{modality}_histogram_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved group histogram comparison -> {out_path}")


# ============================================
# MAIN DRIVER
# ============================================

if __name__ == "__main__":
    logger = setup_logging(OUT_DIR)
    print(f"Normalization method: {NORMALIZATION_METHOD}")

    subjects = find_biascorrected_subjects(BIASCORRECTED_DIR)
    print(f"Found {len(subjects)} bias-corrected subject(s) under BIASCORRECTED_DIR='{BIASCORRECTED_DIR}'")

    selected = load_subject_list()
    subject_items = [(s, p) for s, p in subjects.items() if s in selected]
    if MAX_SUBJECTS is not None:
        subject_items = subject_items[:MAX_SUBJECTS]
    print(f"Processing {len(subject_items)} subject(s) "
          f"(MAX_SUBJECTS={MAX_SUBJECTS} - change at the top of the file to run fewer)")
    logger.info(f"Starting Step 4 on {len(subject_items)} subject(s), OVERWRITE={OVERWRITE}")

    report_path = report_file(os.path.join(OUT_DIR, "step4_normalization_report.csv"))
    previous_report_df = load_previous_report(report_path)

    all_reports = []
    # NOTE: values are stored as float32 (not the float64 used internally
    # during normalization) purely to keep memory bounded when this dict
    # holds every subject's brain-tissue voxels at once for the group-level
    # KS test / histogram at full-dataset scale. Per-subject report stats
    # above are computed from the full-precision values before this cast,
    # so they are unaffected.
    t1_values_by_subject = {}
    flair_values_by_subject = {}

    for sub_id, paths in tqdm(subject_items, desc="Step 4: Normalization"):
        if not OVERWRITE and normalization_outputs_exist(sub_id):
            msg = f"{sub_id}: normalized outputs already exist, skipping computation (OVERWRITE=False)"
            print(f"  {msg}")
            logger.info(msg)
            if not previous_report_df.empty:
                prior_rows = previous_report_df[previous_report_df["sub_id"] == sub_id]
                if not prior_rows.empty:
                    all_reports.extend(prior_rows.to_dict("records"))
                else:
                    logger.warning(f"{sub_id}: outputs exist but no prior report row found - "
                                    f"report will be missing this subject unless OVERWRITE=True")

            sub_out_dir = os.path.join(OUT_DIR, sub_id)
            try:
                t1_values_by_subject[sub_id] = reload_valid_normalized_values(
                    os.path.join(sub_out_dir, f"{sub_id}_T1w_normalized.nii.gz"),
                    os.path.join(sub_out_dir, f"{sub_id}_brain_mask.nii.gz"),
                )
                flair_values_by_subject[sub_id] = reload_valid_normalized_values(
                    os.path.join(sub_out_dir, f"{sub_id}_FLAIR_normalized.nii.gz"),
                    os.path.join(sub_out_dir, f"{sub_id}_brain_mask.nii.gz"),
                )
            except Exception as e:
                logger.error(f"{sub_id}: could not reload cached normalized values for "
                              f"group-level tests: {e}")
            continue

        print(f"\n=== {sub_id} ===")

        try:
            # Mask read once and reused for both T1 and FLAIR normalization
            # calls, and for the final save (previously read 3x per subject).
            mask = sitk.ReadImage(paths["mask"], sitk.sitkUInt8)

            t1_norm_img, t1_info, t1_values = normalize_image(paths["t1"], mask, NORMALIZATION_METHOD, "T1")
            report_t1 = validate_normalization(t1_values, t1_info, sub_id, "T1", NORMALIZATION_METHOD)
            all_reports.append(report_t1)

            flair_norm_img, flair_info, flair_values = normalize_image(paths["flair"], mask, NORMALIZATION_METHOD, "FLAIR")
            report_flair = validate_normalization(flair_values, flair_info, sub_id, "FLAIR", NORMALIZATION_METHOD)
            all_reports.append(report_flair)

            if report_t1["passed"] and report_flair["passed"]:
                sub_out_dir = os.path.join(OUT_DIR, sub_id)
                os.makedirs(sub_out_dir, exist_ok=True)
                sitk.WriteImage(t1_norm_img, os.path.join(sub_out_dir, f"{sub_id}_T1w_normalized.nii.gz"))
                sitk.WriteImage(flair_norm_img, os.path.join(sub_out_dir, f"{sub_id}_FLAIR_normalized.nii.gz"))
                sitk.WriteImage(mask, os.path.join(sub_out_dir, f"{sub_id}_brain_mask.nii.gz"))
                print(f"  saved normalized volumes -> {sub_out_dir}")

                t1_values_by_subject[sub_id] = t1_values.astype(np.float32)
                flair_values_by_subject[sub_id] = flair_values.astype(np.float32)
            else:
                msg = f"{sub_id}: too many extreme z-score voxels, NOT saved - inspect this subject"
                logger.warning(msg)

        except Exception as e:
            msg = f"Subject {sub_id} failed during normalization: {e}"
            logger.error(msg)

        # Write the report after every subject so an interrupted long
        # batch run doesn't lose already-completed work.
        pd.DataFrame(all_reports).to_csv(report_path, index=False)

    report_df = pd.DataFrame(all_reports)
    if report_df.empty:  # e.g. a parallel shard whose subjects were all already done
        print("No subjects processed or reloaded in this run - nothing to summarise.")
        raise SystemExit(0)
    report_df.to_csv(report_path, index=False)
    print(f"\nSaved per-subject normalization report -> {report_path}")
    print(report_df)

    print("\nChecking for peak outlier subjects (possible upstream anomalies)...")
    for modality in ["T1", "FLAIR"]:
        flag_peak_outliers(report_df, modality)

    for modality, values_by_subject in [("T1", t1_values_by_subject), ("FLAIR", flair_values_by_subject)]:
        if len(values_by_subject) < 2:
            print(f"\nNot enough subjects for group-level {modality} checks.")
            continue

        ks_df = group_level_ks_test(values_by_subject, modality)
        ks_path = os.path.join(OUT_DIR, f"step4_{modality}_ks_tests.csv")
        ks_df.to_csv(ks_path, index=False)

        n_flagged = ks_df["flagged_different"].sum()
        n_total = len(ks_df)
        mean_d = ks_df["ks_stat"].mean()
        print(f"\n{modality} group-level KS tests (subsampled to {CONFIG['ks_subsample_n']} voxels/subject): "
              f"{n_total} pairs compared")
        print(f"  {n_flagged}/{n_total} = {100*n_flagged/n_total:.1f}% flagged "
              f"(p<{CONFIG['ks_pvalue_flag_threshold']} AND D>{CONFIG['ks_statistic_flag_threshold']})")
        print(f"  mean KS statistic (D) across all pairs: {mean_d:.4f} (lower = more similar distributions)")
        print(f"  saved -> {ks_path}")

        save_histogram_comparison(values_by_subject, modality, NORMALIZATION_METHOD)
