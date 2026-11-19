# ============================================
# STEP 2: SKULL STRIPPING
# ============================================
# Standalone script. Reads the REGISTERED outputs from Step 1
# (registered_outputs/{sub_id}/{sub_id}_T1w.nii.gz and
#  registered_outputs/{sub_id}/{sub_id}_FLAIR_in_T1space.nii.gz),
# so make sure Step 1 has already run and saved those before running this.
#
# Install once (same as Step 1, plus HD-BET):
#   pip install SimpleITK nibabel nilearn scipy numpy matplotlib pandas tqdm hd-bet
#
# APPROACH: HD-BET (Isensee et al. 2019), a deep-learning brain extraction
# tool trained on multi-centre clinical MRI, run on the T1. Since T1 and
# FLAIR are already registered to the same grid (Step 1), we compute the
# mask ONCE on T1 and apply the SAME mask to FLAIR - this is more
# consistent than computing independent masks on each modality (which is
# exactly what caused the Dice confusion in Step 1's validation).
#
# WHY NOT nilearn.compute_brain_mask (the previous method): it does NOT
# segment the subject's brain. It resamples a fixed MNI152 template mask
# onto the image grid, assuming the image is already in MNI space. These
# T1s are in native scanner space, so the template only roughly overlapped
# each brain: it cut off the superior cortex, included posterior skull and
# neck, gave near-identical volumes for men and women, and (via head
# position/FOV differences between the patient and control acquisitions)
# gave controls ~64 cm3 larger masks than patients (p=0.002) - a scanner
# confound the classifier could have learned. HD-BET masks overlapped the
# template masks with Dice of only ~0.5-0.6 on the subjects checked.
#
# What this does, per subject:
#   1. Runs HD-BET on the registered T1 -> binary brain mask -> keeps only
#      the largest connected component (safety net against stray blobs)
#   2. Validates the mask against an expected whole-brain volume range
#   3. Validates mask STABILITY (erode-then-dilate should reproduce the
#      same mask almost exactly - if not, the mask has noisy/ragged edges)
#   4. Applies the SAME mask to the registered FLAIR (no separate
#      extraction on FLAIR - avoids the modality-mismatch problem from Step 1)
#   5. Saves skull-stripped T1 + FLAIR + the mask itself, plus QC figures

import os
import glob
import logging
import subprocess

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk

from nilearn import plotting
from scipy import stats
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================

CONFIG = {
    # Plausible adult brain-mask volume range (brain tissue + the CSF
    # HD-BET includes), in cm3. Healthy adults span roughly 1000-1700;
    # the limits are deliberately loose so only a clearly broken mask
    # (half a brain, or skull included) fails. The two HD-BET test subjects
    # came out at 1395 and 1467 cm3. Max raised 2000 -> 2200 after sub-132
    # (male, 45-49) measured 2143 cm3: its mask was checked visually and is
    # brain-only, and its voxel sizes match the scanner JSON - a genuinely
    # large head, not a skull-stripping error.
    "brain_volume_min_cm3": 900,
    "brain_volume_max_cm3": 2200,

    # Mask stability: Dice between the mask and an eroded-then-dilated
    # version of itself. Low stability = ragged/noisy mask edges.
    "stability_dice_threshold": 0.95,
}

# HD-BET runs on CPU on the laptop (no CUDA GPU); set TLE_HDBET_DEVICE=cuda
# on the GPU server. Test-time augmentation stays disabled everywhere so
# masks are produced the same way on both machines (~40 s/subject on CPU).
HDBET_DEVICE = os.environ.get("TLE_HDBET_DEVICE", "cpu")  # "cuda" on the GPU server
HDBET_DISABLE_TTA = True

MAX_SUBJECTS = None  # None = process everyone found in REGISTERED_DIR

# If a subject's expected output files already exist, skip recomputing
# them. Set True to force reprocessing everyone.
OVERWRITE = False

from pipeline_config import LABELS_FILE, RUN_DIR, load_subject_list, report_file

# Where Step 1's outputs live (read-only input to this step)
REGISTERED_DIR = os.path.join(RUN_DIR, "registered_outputs")

# Where this step's outputs get saved
OUT_DIR = os.path.join(RUN_DIR, "skullstripped_outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def setup_logging(out_dir):
    logger = logging.getLogger("step2_skull_stripping")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(os.path.join(out_dir, "step2_skull_stripping.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


logger = logging.getLogger("step2_skull_stripping")


# ============================================
# DISCOVER SUBJECTS FROM STEP 1's OUTPUT
# ============================================

def find_registered_subjects(registered_dir):
    """Finds subjects that Step 1 successfully registered and saved."""
    subject_dirs = sorted(
        d for d in glob.glob(os.path.join(registered_dir, "sub-*")) if os.path.isdir(d)
    )
    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject folders found under REGISTERED_DIR='{registered_dir}'. "
            f"Make sure Step 1 (step1_registration.py) has been run first."
        )

    subjects = {}
    for sub_dir in subject_dirs:
        sub_id = os.path.basename(sub_dir)
        t1_path = os.path.join(sub_dir, f"{sub_id}_T1w.nii.gz")
        flair_path = os.path.join(sub_dir, f"{sub_id}_FLAIR_in_T1space.nii.gz")

        if os.path.exists(t1_path) and os.path.exists(flair_path):
            subjects[sub_id] = {"t1": t1_path, "flair": flair_path}
        else:
            msg = f"{sub_id}: missing expected files in {sub_dir}, skipping."
            print(msg)
            logger.warning(msg)

    return subjects


def skullstrip_outputs_exist(sub_id):
    sub_out_dir = os.path.join(OUT_DIR, sub_id)
    return all(
        os.path.exists(os.path.join(sub_out_dir, f"{sub_id}{suffix}"))
        for suffix in ["_T1w_brain.nii.gz", "_FLAIR_brain.nii.gz", "_brain_mask.nii.gz"]
    )


def load_previous_report(report_path):
    if os.path.exists(report_path):
        try:
            return pd.read_csv(report_path)
        except Exception as e:
            logger.warning(f"Could not read previous report at {report_path}: {e}")
    return pd.DataFrame()


# ============================================
# CORE SKULL STRIPPING
# ============================================

def run_hdbet(t1_path, sub_id):
    """Runs the HD-BET command-line tool on one T1 and returns the path of
    the binary mask it writes. (Plain Otsu thresholding was tried first and
    can't separate brain from equally-bright scalp fat / skull marrow at the
    skull base, right next to the temporal lobes this study cares about.)"""
    # One temp folder per subject: HD-BET writes and deletes helper files
    # (dataset.json) in its output folder, so parallel shards sharing one
    # folder deleted each other's files.
    tmp_dir = os.path.join(OUT_DIR, "_hdbet_tmp", sub_id)
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, f"{sub_id}_brain.nii.gz")
    cmd = ["hd-bet", "-i", t1_path, "-o", out_path, "-device", HDBET_DEVICE,
           "--save_bet_mask", "--no_bet_image"]
    if HDBET_DISABLE_TTA:
        cmd.append("--disable_tta")
    result = subprocess.run(cmd, capture_output=True, text=True)
    mask_path = out_path.replace(".nii.gz", "_bet.nii.gz")
    if result.returncode != 0 or not os.path.exists(mask_path):
        raise RuntimeError(f"HD-BET failed for {sub_id}: {result.stderr.strip()[-500:]}")
    return mask_path


def skull_strip_hdbet(t1_path, sub_id):
    """Brain extraction with HD-BET (see header). Returns the skull-stripped
    T1 and the binary mask, both on the T1's own voxel grid."""
    sitk_img = sitk.ReadImage(t1_path, sitk.sitkFloat32)

    mask_path = run_hdbet(t1_path, sub_id)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    os.remove(mask_path)
    if mask.GetSize() != sitk_img.GetSize():
        raise RuntimeError(f"HD-BET mask grid {mask.GetSize()} != T1 grid {sitk_img.GetSize()}")
    # HD-BET writes the mask on the input grid; copy the geometry exactly
    # so sitk.Mask never rejects it over floating-point header differences.
    mask.CopyInformation(sitk_img)
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

    cc = sitk.ConnectedComponent(mask)
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(cc)
    if len(shape_stats.GetLabels()) == 0:
        raise RuntimeError(f"Skull stripping found no components for {t1_path}")
    largest_label = max(shape_stats.GetLabels(), key=lambda l: shape_stats.GetPhysicalSize(l))
    mask = sitk.Equal(cc, largest_label)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    brain = sitk.Mask(sitk_img, mask)
    return brain, mask



def dice_coefficient(mask_a, mask_b):
    a = sitk.GetArrayFromImage(mask_a).astype(bool)
    b = sitk.GetArrayFromImage(mask_b).astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / denom


# ============================================
# VALIDATION
# ============================================

def validate_skull_strip(mask, sub_id):
    spacing = mask.GetSpacing()
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    volume_cm3 = (mask_arr.sum() * voxel_vol_mm3) / 1000.0
    volume_pass = CONFIG["brain_volume_min_cm3"] <= volume_cm3 <= CONFIG["brain_volume_max_cm3"]

    eroded = sitk.BinaryErode(mask, [2, 2, 2])
    reconstructed = sitk.BinaryDilate(eroded, [2, 2, 2])
    stability_dice = dice_coefficient(mask, reconstructed)
    stability_pass = stability_dice >= CONFIG["stability_dice_threshold"]

    overall = volume_pass and stability_pass
    status = "PASS" if overall else "FAIL"

    print(f"[{status}] Skull stripping - {sub_id} | "
          f"volume={volume_cm3:.1f} cm3 "
          f"(need {CONFIG['brain_volume_min_cm3']}-{CONFIG['brain_volume_max_cm3']}) | "
          f"stability Dice={stability_dice:.3f} (need >= {CONFIG['stability_dice_threshold']})")

    return {
        "sub_id": sub_id, "volume_cm3": volume_cm3,
        "stability_dice": stability_dice, "passed": overall,
    }


# ============================================
# QC VISUALIZATION
# ============================================

def get_mask_coronal_cut_coords(mask_path, n_cuts=6, margin_fraction=0.15):
    """Picks n_cuts coronal (y-axis) world coordinates spread evenly across
    the ACTUAL mask extent (not the whole image, which can include a lot
    of empty background and waste slices on nothing, as happened with
    nilearn's automatic picker). Uses nibabel + the image's own affine so
    the coordinate convention matches exactly what nilearn expects for
    cut_coords - no LPS/RAS ambiguity from mixing libraries."""
    img = nib.load(mask_path)
    data = np.asarray(img.get_fdata())
    affine = img.affine

    nz = np.array(np.nonzero(data))
    if nz.shape[1] == 0:
        return None

    i_min, j_min, k_min = nz.min(axis=1)
    i_max, j_max, k_max = nz.max(axis=1)
    i_center = (i_min + i_max) / 2.0
    k_center = (k_min + k_max) / 2.0

    j_range = j_max - j_min
    j_lo = j_min + margin_fraction * j_range
    j_hi = j_max - margin_fraction * j_range
    js = np.linspace(j_lo, j_hi, n_cuts)

    world_ys = [
        nib.affines.apply_affine(affine, [i_center, j, k_center])[1] for j in js
    ]
    return sorted(world_ys)


def save_qc_figures(t1_path, mask, brain, sub_id):
    mask_path = os.path.join(OUT_DIR, f"_tmp_{sub_id}_mask.nii.gz")
    sitk.WriteImage(mask, mask_path)

    # mask overlaid on native T1 - shows what got included/excluded
    display = plotting.plot_roi(
        mask_path, bg_img=t1_path, title=f"{sub_id} brain mask", cmap="autumn", alpha=0.5
    )
    overlay_path = os.path.join(FIG_DIR, f"{sub_id}_mask_overlay.png")
    display.savefig(overlay_path)
    display.close()

    # skull-stripped result on its own - shows if any brain got cut off
    brain_path = os.path.join(OUT_DIR, f"_tmp_{sub_id}_brain.nii.gz")
    sitk.WriteImage(brain, brain_path)
    display = plotting.plot_anat(brain_path, title=f"{sub_id} skull-stripped T1", display_mode="ortho")
    stripped_path = os.path.join(FIG_DIR, f"{sub_id}_stripped_result.png")
    display.savefig(stripped_path)
    display.close()

    # Coronal series - slices spread evenly across the ACTUAL mask extent
    # (computed above), so every slice shows real brain tissue instead of
    # wasting slices on empty background.
    cut_coords = get_mask_coronal_cut_coords(mask_path, n_cuts=6)
    display = plotting.plot_roi(
        mask_path, bg_img=t1_path, title=f"{sub_id} mask - coronal series",
        cmap="autumn", alpha=0.5, display_mode="y", cut_coords=cut_coords
    )
    temporal_path = os.path.join(FIG_DIR, f"{sub_id}_temporal_lobe_check.png")
    display.savefig(temporal_path)
    display.close()

    print(f"  saved QC figures -> {overlay_path}, {stripped_path}, {temporal_path}")


# ============================================
# MAIN DRIVER
# ============================================

if __name__ == "__main__":
    logger = setup_logging(OUT_DIR)

    subjects = find_registered_subjects(REGISTERED_DIR)
    print(f"Found {len(subjects)} registered subject(s) under REGISTERED_DIR='{REGISTERED_DIR}'")

    selected = load_subject_list()
    subject_items = [(s, p) for s, p in subjects.items() if s in selected]
    if MAX_SUBJECTS is not None:
        subject_items = subject_items[:MAX_SUBJECTS]
    print(f"Processing {len(subject_items)} subject(s) "
          f"(MAX_SUBJECTS={MAX_SUBJECTS} - change at the top of the file to run fewer)")
    logger.info(f"Starting Step 2 on {len(subject_items)} subject(s), OVERWRITE={OVERWRITE}")

    report_path = report_file(os.path.join(OUT_DIR, "step2_skullstrip_report.csv"))
    previous_report_df = load_previous_report(report_path)

    all_reports = []

    for sub_id, paths in tqdm(subject_items, desc="Step 2: Skull stripping"):
        if not OVERWRITE and skullstrip_outputs_exist(sub_id):
            msg = f"{sub_id}: skull-stripped outputs already exist, skipping (OVERWRITE=False)"
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

        print(f"\n=== {sub_id} ===")

        try:
            brain, mask = skull_strip_hdbet(paths["t1"], sub_id)
            report = validate_skull_strip(mask, sub_id)
            all_reports.append(report)
            save_qc_figures(paths["t1"], mask, brain, sub_id)

            if report["passed"]:
                # apply the SAME mask to FLAIR - no separate Otsu run on
                # FLAIR, this keeps masks consistent across modalities
                flair_img = sitk.ReadImage(paths["flair"], sitk.sitkFloat32)
                flair_brain = sitk.Mask(flair_img, mask)

                sub_out_dir = os.path.join(OUT_DIR, sub_id)
                os.makedirs(sub_out_dir, exist_ok=True)
                sitk.WriteImage(brain, os.path.join(sub_out_dir, f"{sub_id}_T1w_brain.nii.gz"))
                sitk.WriteImage(flair_brain, os.path.join(sub_out_dir, f"{sub_id}_FLAIR_brain.nii.gz"))
                sitk.WriteImage(mask, os.path.join(sub_out_dir, f"{sub_id}_brain_mask.nii.gz"))
                print(f"  saved skull-stripped volumes -> {sub_out_dir}")
            else:
                msg = f"{sub_id}: skull stripping did not pass validation, NOT saved for downstream use"
                logger.warning(msg)

        except Exception as e:
            msg = f"Subject {sub_id} failed during skull stripping: {e}"
            logger.error(msg)

        # Write the report after every subject so an interrupted long
        # batch run doesn't lose already-completed work.
        pd.DataFrame(all_reports).to_csv(report_path, index=False)

    report_df = pd.DataFrame(all_reports)
    if report_df.empty:  # e.g. a parallel shard whose subjects were all already done
        print("No subjects processed or reloaded in this run - nothing to summarise.")
        raise SystemExit(0)
    report_df.to_csv(report_path, index=False)
    print(f"\nSaved skull-stripping validation report -> {report_path}")
    print(report_df)

    if len(report_df) > 0:
        pass_rate = report_df["passed"].mean() * 100
        print(f"\nPass rate: {pass_rate:.1f}% ({report_df['passed'].sum()}/{len(report_df)} subjects)")
        print(f"Mean brain volume: {report_df['volume_cm3'].mean():.1f} cm3 "
              f"(std: {report_df['volume_cm3'].std():.1f})")

        # Confound check: mask volume by group and sex. Real brains differ
        # by sex (~10% larger in men) and only slightly between TLE patients
        # and controls. The old template-based masks showed the reverse -
        # no sex difference but a large group difference - which is how
        # that method's failure was caught.
        labels = pd.read_csv(LABELS_FILE, sep="\t")
        if "cohort" in labels.columns:  # group comparison only makes sense on matched pairs
            labels = labels[labels["cohort"] == "matched"]
        merged = report_df.merge(labels, left_on="sub_id", right_on="participant_id")
        print("\nMean mask volume (cm3) by group and sex:")
        print(merged.groupby(["group", "sex"])["volume_cm3"].agg(["mean", "std", "count"]).round(1))
        groups = [g["volume_cm3"].values for _, g in merged.groupby("group")]
        sexes = [g["volume_cm3"].values for _, g in merged.groupby("sex")]
        if len(groups) == 2 and len(sexes) == 2:
            print(f"Mann-Whitney patient vs control: p={stats.mannwhitneyu(*groups).pvalue:.4f}")
            print(f"Mann-Whitney F vs M:             p={stats.mannwhitneyu(*sexes).pvalue:.4f}")
