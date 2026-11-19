# ============================================
# STEP 1: REGISTRATION (T1 <-> FLAIR)
# ============================================
# Standalone script. Run this on its own before moving to Step 2.
#
# Install once:
#   pip install SimpleITK nibabel nilearn scipy numpy matplotlib pandas tqdm
#
# What this does, per subject:
#   1. Registers FLAIR -> T1 (affine, Mattes Mutual Information)
#   2. Registers T1 -> FLAIR (same, other direction)
#   3. For each direction: computes MI BEFORE registration (native, unaligned)
#      and AFTER registration, plus a rough Dice overlap of brain masks
#   4. Saves a checkerboard visualization so you can eyeball alignment
#   5. Across all subjects: runs a paired statistical test (Wilcoxon
#      signed-rank, since MI values are rarely normally distributed) on
#      MI-before vs MI-after, to prove the improvement isn't a fluke

import os
import glob
import logging

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from nilearn import plotting
from scipy import stats
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================

CONFIG = {
    # MI is negative; more negative = better. This is the PRIMARY and ONLY
    # pass/fail criterion for registration - it's a reliable, direct
    # measure of alignment quality.
    "registration_mi_threshold": -0.35,

    # NOTE ON DICE: an Otsu-mask Dice overlap was tried as a second
    # validation metric and found to be UNRELIABLE for this dataset -
    # confirmed by visual inspection (checkerboards showed correct
    # alignment even when Dice was as low as 0.16). Root cause: FLAIR
    # in this dataset is acquired at an oblique angle (common in epilepsy
    # protocols, to better visualize the hippocampus) with a very
    # different intensity range/contrast than T1 (FLAIR suppresses CSF
    # signal). Independent Otsu thresholding on each modality produces
    # masks that don't overlap well even when the true anatomical
    # alignment is correct. Kept here for reference only, NOT used as
    # a pass/fail gate anywhere in this script.
    "registration_dice_reference_only": True,
}

# Fixed seed for MI's random voxel sampling. Without it every run samples
# different voxels, so a borderline subject (MI near the -0.35 threshold)
# can pass on one run and fail on the next.
SAMPLING_SEED = 42

# How many subjects to process. Set to None to run the FULL dataset.
MAX_SUBJECTS = None

# If a subject's expected output files already exist, skip recomputing
# them. Set True to force reprocessing everyone (e.g. after a real
# algorithm/threshold change).
OVERWRITE = False

from pipeline_config import DATA_ROOT, RUN_DIR, load_subject_list, report_file

OUT_DIR = os.path.join(RUN_DIR, "step1_registration_outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Registered NIfTI volumes get saved here, one subfolder per subject, so
# Step 2 (skull stripping) and beyond can load them directly instead of
# re-running registration every time.
REGISTERED_DIR = os.path.join(RUN_DIR, "registered_outputs")
os.makedirs(REGISTERED_DIR, exist_ok=True)


def setup_logging(out_dir):
    """Persistent log file alongside the existing console output, so a
    long batch run over the full dataset leaves a record of what was
    skipped/failed and why, even if the terminal scrollback is gone."""
    logger = logging.getLogger("step1_registration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(os.path.join(out_dir, "step1_registration.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


logger = logging.getLogger("step1_registration")


def find_subject_files(data_root):
    """Map each subject to its exact T1w/FLAIR paths using the BIDS folder
    structure directly (sub-X/anat/sub-X_T1w.nii.gz), instead of substring
    matching on a flat file list - substring matching is unsafe here since
    e.g. "sub-1" is a substring of "sub-10", "sub-100", "sub-11", etc."""
    subject_dirs = sorted(
        d for d in glob.glob(os.path.join(data_root, "sub-*")) if os.path.isdir(d)
    )
    if not subject_dirs:
        raise FileNotFoundError(
            f"No sub-* folders found under DATA_ROOT='{data_root}'. "
            f"Check DATA_ROOT matches where you're running this script from."
        )

    t1_map, flair_map = {}, {}
    for sub_dir in subject_dirs:
        sub_id = os.path.basename(sub_dir)
        anat_dir = os.path.join(sub_dir, "anat")

        t1_path = os.path.join(anat_dir, f"{sub_id}_T1w.nii.gz")
        flair_path = os.path.join(anat_dir, f"{sub_id}_FLAIR.nii.gz")

        if os.path.exists(t1_path):
            t1_map[sub_id] = t1_path
        if os.path.exists(flair_path):
            flair_map[sub_id] = flair_path

    return t1_map, flair_map


def registration_outputs_exist(sub_id):
    """True if this subject's final Step 1 outputs are already on disk."""
    sub_out_dir = os.path.join(REGISTERED_DIR, sub_id)
    t1_out = os.path.join(sub_out_dir, f"{sub_id}_T1w.nii.gz")
    flair_out = os.path.join(sub_out_dir, f"{sub_id}_FLAIR_in_T1space.nii.gz")
    return os.path.exists(t1_out) and os.path.exists(flair_out)


def load_previous_report(report_path):
    """For resuming an interrupted run: prior per-subject rows so skipped
    (already-done) subjects still contribute to the final report and the
    group-level statistical test."""
    if os.path.exists(report_path):
        try:
            return pd.read_csv(report_path)
        except Exception as e:
            logger.warning(f"Could not read previous report at {report_path}: {e}")
    return pd.DataFrame()


# ============================================
# UTILITIES
# ============================================

def rough_brain_mask(sitk_img):
    """Quick Otsu mask - just for a lightweight registration sanity check,
    NOT your final skull-stripping mask (that's Step 2)."""
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    return otsu.Execute(sitk.Cast(sitk_img, sitk.sitkFloat32))


def dice_coefficient(mask_a, mask_b):
    a = sitk.GetArrayFromImage(mask_a).astype(bool)
    b = sitk.GetArrayFromImage(mask_b).astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / denom


def mattes_mi(fixed, moving, transform):
    """Evaluate Mattes MI for a given transform, without optimizing.
    Used to measure MI BEFORE registration (identity/initial transform)
    so we have a real baseline to compare against."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, SAMPLING_SEED)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(transform, inPlace=False)
    # MetricEvaluate computes the metric at the initial transform directly.
    # (The previous version ran Execute() with 0 optimizer iterations and
    # read GetMetricValue(), which never evaluates the metric and returned
    # exactly 0.0 for every subject - making the before/after Wilcoxon test
    # compare against a fake baseline.)
    return reg.MetricEvaluate(fixed, moving)


def make_checkerboard(fixed, moving_resampled, num_checkers=8):
    checker_filter = sitk.CheckerBoardImageFilter()
    checker_filter.SetCheckerPattern([num_checkers] * fixed.GetDimension())
    return checker_filter.Execute(fixed, moving_resampled)


# ============================================
# CORE REGISTRATION FUNCTION
# ============================================

def _run_single_registration(fixed, moving, init_method):
    """One registration attempt from one starting position."""
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3), init_method
    )
    mi_before = mattes_mi(fixed, moving, initial_transform)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, SAMPLING_SEED)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_transform, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = reg.Execute(fixed, moving)
    mi_after = reg.GetMetricValue()

    # Some IDEAS pairs are already near-perfectly aligned before any
    # optimisation (e.g. sub-237: MI -0.505 at the start). There the
    # optimiser can only drift, and it made MI slightly worse (-0.480), which
    # the "improved" gate then rejected - discarding one of the best-aligned
    # subjects. So: if the optimised transform is worse than where it
    # started (measured the same way as mi_before), keep the start.
    if mattes_mi(fixed, moving, final_transform) > mi_before:
        logger.info(f"  optimisation did not improve MI from {mi_before:.4f}; "
                    f"keeping the initial (already aligned) transform")
        return initial_transform, mi_before, mi_before
    return final_transform, mi_before, mi_after


def register_images(fixed, moving):
    """Affine-registers `moving` onto `fixed` (both already-loaded sitk
    images - read once per subject in the driver and passed in, instead
    of each direction re-reading the same two files from disk). Call
    twice with fixed/moving swapped to get both directions (T1<->FLAIR).

    Affine registration with a single starting guess can converge to a
    LOCAL optimum that scores okay on MI but is visibly wrong (brains are
    roughly symmetric, so this happens more than you'd expect). To guard
    against that, this tries two different starting positions - GEOMETRY
    (aligns image bounding-box centers) and MOMENTS (aligns intensity-
    weighted centers of mass, more robust when T1/FLAIR have different
    field-of-view coverage) - and keeps whichever one actually converges
    to the better (more negative) final MI."""
    candidates = []
    for init_method, name in [
        (sitk.CenteredTransformInitializerFilter.GEOMETRY, "geometry"),
        (sitk.CenteredTransformInitializerFilter.MOMENTS, "moments"),
    ]:
        try:
            transform, mi_before, mi_after = _run_single_registration(fixed, moving, init_method)
            candidates.append((mi_after, mi_before, transform, name))
        except Exception as e:
            logger.warning(f"  registration attempt with '{name}' init failed: {e}")

    if not candidates:
        raise RuntimeError("All registration attempts failed for this fixed/moving pair")

    # more negative MI = better; pick the winner
    candidates.sort(key=lambda c: c[0])
    mi_after, mi_before, final_transform, winning_init = candidates[0]
    print(f"  best init: {winning_init} (MI after={mi_after:.4f})")

    moving_resampled = sitk.Resample(
        moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    return moving_resampled, final_transform, mi_before, mi_after, fixed


# ============================================
# VALIDATION (per subject, per direction)
# ============================================

def validate_registration(fixed_img, registered_moving_img, mi_before, mi_after, sub_id, direction):
    # more negative = better; "equal" happens when the initial transform was
    # kept because optimisation couldn't improve on it (see
    # _run_single_registration) - that's not a failure
    improved = mi_after <= mi_before
    threshold_pass = mi_after <= CONFIG["registration_mi_threshold"]
    overall = improved and threshold_pass  # MI is the sole gate

    fixed_mask = rough_brain_mask(fixed_img)
    moving_mask = rough_brain_mask(registered_moving_img)
    dice = dice_coefficient(fixed_mask, moving_mask)  # informational only

    status = "PASS" if overall else "FAIL"

    print(f"[{status}] Registration ({direction}) - {sub_id} | "
          f"MI before={mi_before:.4f} after={mi_after:.4f} "
          f"(improved={improved}, threshold<= {CONFIG['registration_mi_threshold']}) | "
          f"Dice(info only)={dice:.3f}")

    return {
        "sub_id": sub_id, "direction": direction,
        "mi_before": mi_before, "mi_after": mi_after,
        "dice": dice, "passed": overall,
    }


def save_qc_figures(fixed_img, registered_moving_img, sub_id, direction):
    checkerboard = make_checkerboard(fixed_img, registered_moving_img)
    checker_path = os.path.join(OUT_DIR, f"_tmp_{sub_id}_{direction}_checker.nii.gz")
    sitk.WriteImage(checkerboard, checker_path)

    display = plotting.plot_anat(
        checker_path, title=f"{sub_id} checkerboard ({direction})", display_mode="ortho"
    )
    out_path = os.path.join(FIG_DIR, f"{sub_id}_{direction}_checkerboard.png")
    display.savefig(out_path)
    display.close()
    print(f"  saved checkerboard QC figure -> {out_path}")


def save_raw_comparison(t1_path, flair_path, sub_id):
    """Saves the raw, UNREGISTERED T1 and FLAIR as separate figures so you
    can check for an underlying issue (different field-of-view, cropping,
    orientation) before registration even happens - useful when a subject's
    registration looks visually wrong despite a decent MI score."""
    t1_out = os.path.join(FIG_DIR, f"{sub_id}_raw_T1.png")
    display = plotting.plot_anat(t1_path, title=f"{sub_id} raw T1 (native)", display_mode="ortho")
    display.savefig(t1_out)
    display.close()

    flair_out = os.path.join(FIG_DIR, f"{sub_id}_raw_FLAIR.png")
    display = plotting.plot_anat(flair_path, title=f"{sub_id} raw FLAIR (native)", display_mode="ortho")
    display.savefig(flair_out)
    display.close()

    print(f"  saved raw comparison figures -> {t1_out}, {flair_out}")


# ============================================
# GROUP-LEVEL STATISTICAL VALIDATION
# ============================================

def group_level_test(report_df, direction):
    subset = report_df[report_df["direction"] == direction]
    if len(subset) < 2:
        print(f"Not enough subjects to run a group-level test for {direction} "
              f"(need >= 2, have {len(subset)}).")
        return None

    before = subset["mi_before"].values
    after = subset["mi_after"].values

    # Wilcoxon signed-rank: paired, doesn't assume normality - appropriate
    # for a handful of subjects with metric values that may be skewed.
    stat, p_value = stats.wilcoxon(before, after, alternative="greater")
    # alternative="greater" tests H1: before > after (i.e. after is more
    # negative / better), matching "registration improves alignment"

    print(f"\n{direction} group-level Wilcoxon signed-rank test:")
    print(f"  n={len(subset)}, statistic={stat:.4f}, p-value={p_value:.5f}")
    if p_value < 0.05:
        print("  -> Significant improvement in alignment after registration (p < 0.05)")
    else:
        print("  -> NOT significant - inspect individual subjects / QC figures")

    return {"direction": direction, "n": len(subset), "statistic": stat, "p_value": p_value}


# ============================================
# MAIN DRIVER
# ============================================

if __name__ == "__main__":
    logger = setup_logging(OUT_DIR)

    t1_map, flair_map = find_subject_files(DATA_ROOT)
    print(f"Found {len(t1_map)} T1w and {len(flair_map)} FLAIR files under DATA_ROOT='{DATA_ROOT}'")

    # Only the cohort chosen by Step 0 - not every sub-* folder on disk.
    selected = load_subject_list()
    missing = sorted(selected - set(t1_map))
    if missing:
        logger.warning(f"{len(missing)} selected subject(s) have no T1w on disk: {missing}")
    subjects_to_run = [(s, p) for s, p in t1_map.items() if s in selected]
    if MAX_SUBJECTS is not None:
        subjects_to_run = subjects_to_run[:MAX_SUBJECTS]
    print(f"Processing {len(subjects_to_run)} subject(s) "
          f"(MAX_SUBJECTS={MAX_SUBJECTS} - change this at the top of the file to run fewer)")
    logger.info(f"Starting Step 1 on {len(subjects_to_run)} subject(s), OVERWRITE={OVERWRITE}")

    report_path = report_file(os.path.join(OUT_DIR, "step1_registration_report.csv"))
    previous_report_df = load_previous_report(report_path)

    all_reports = []

    for sub_id, t1_path in tqdm(subjects_to_run, desc="Step 1: Registration"):
        if sub_id not in flair_map:
            msg = f"{sub_id}: no FLAIR found, skipping."
            print(msg)
            logger.warning(msg)
            continue

        if not OVERWRITE and registration_outputs_exist(sub_id):
            msg = f"{sub_id}: registered outputs already exist, skipping (OVERWRITE=False)"
            print(f"  {msg}")
            logger.info(msg)
            if not previous_report_df.empty:
                prior_rows = previous_report_df[previous_report_df["sub_id"] == sub_id]
                if not prior_rows.empty:
                    all_reports.extend(prior_rows.to_dict("records"))
                else:
                    logger.warning(f"{sub_id}: outputs exist but no prior report row found - "
                                    f"report will be missing this subject unless OVERWRITE=True")
            continue

        flair_path = flair_map[sub_id]

        print(f"\n=== {sub_id} ===")
        save_raw_comparison(t1_path, flair_path, sub_id)

        try:
            # Read each volume once per subject and reuse for both
            # registration directions (previously each direction re-read
            # both files from disk independently).
            t1_img = sitk.ReadImage(t1_path, sitk.sitkFloat32)
            flair_img = sitk.ReadImage(flair_path, sitk.sitkFloat32)

            # Direction 1: FLAIR -> T1
            flair_on_t1, _, mi_b1, mi_a1, t1_fixed = register_images(t1_img, flair_img)
            report1 = validate_registration(t1_fixed, flair_on_t1, mi_b1, mi_a1, sub_id, "FLAIR_to_T1")
            save_qc_figures(t1_fixed, flair_on_t1, sub_id, "FLAIR_to_T1")
            all_reports.append(report1)

            # Direction 2: T1 -> FLAIR
            t1_on_flair, _, mi_b2, mi_a2, flair_fixed = register_images(flair_img, t1_img)
            report2 = validate_registration(flair_fixed, t1_on_flair, mi_b2, mi_a2, sub_id, "T1_to_FLAIR")
            save_qc_figures(flair_fixed, t1_on_flair, sub_id, "T1_to_FLAIR")
            all_reports.append(report2)

            # Save registered volumes to disk for Step 2 onwards - only
            # if both directions passed, so Step 2 never has to guess
            # whether a saved file is trustworthy.
            if report1["passed"] and report2["passed"]:
                sub_out_dir = os.path.join(REGISTERED_DIR, sub_id)
                os.makedirs(sub_out_dir, exist_ok=True)

                # native T1 (unchanged, this is our reference space)
                sitk.WriteImage(t1_fixed, os.path.join(sub_out_dir, f"{sub_id}_T1w.nii.gz"))
                # FLAIR resampled into T1 space (the useful registered output)
                sitk.WriteImage(flair_on_t1, os.path.join(sub_out_dir, f"{sub_id}_FLAIR_in_T1space.nii.gz"))

                print(f"  saved registered volumes -> {sub_out_dir}")
            else:
                msg = f"{sub_id}: registration did not pass validation, NOT saved for downstream use"
                logger.warning(msg)

        except Exception as e:
            msg = f"Subject {sub_id} failed during registration: {e}"
            logger.error(msg)

        # Write the report after every subject (not just at the end) so an
        # interrupted long batch run doesn't lose already-completed work.
        pd.DataFrame(all_reports).to_csv(report_path, index=False)

    report_df = pd.DataFrame(all_reports)
    if report_df.empty:  # e.g. a parallel shard whose subjects were all already done
        print("No subjects processed or reloaded in this run - nothing to summarise.")
        raise SystemExit(0)
    report_df.to_csv(report_path, index=False)
    print(f"\nSaved per-subject registration report -> {report_path}")
    print(report_df)

    # Group-level statistical validation, one test per direction
    group_results = []
    for direction in ["FLAIR_to_T1", "T1_to_FLAIR"]:
        result = group_level_test(report_df, direction)
        if result:
            group_results.append(result)

    if group_results:
        group_df = pd.DataFrame(group_results)
        group_path = os.path.join(OUT_DIR, "step1_group_statistical_test.csv")
        group_df.to_csv(group_path, index=False)
        print(f"\nSaved group-level statistical test results -> {group_path}")
