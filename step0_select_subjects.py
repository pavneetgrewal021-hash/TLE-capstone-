# ============================================
# STEP 0: BALANCED, SCANNER-MATCHED SUBJECT SELECTION
# ============================================
# Run this before Step 1. It decides which subjects the whole pipeline
# runs on and writes their labels, so every later step (and the Step 12
# classifier) uses the same fixed, reproducible cohort.
#
# WHY SCANNER MATCHING: IDEAS subjects were scanned on three GE 3T
# scanners, and the controls were acquired almost entirely on one of them:
#
#                        controls  temporal patients   (with T1w+FLAIR)
#     DISCOVERY MR750       68            92
#     SIGNA HDx             15           163
#     SIGNA EXCITE           0            78
#
# The first balanced cohort (age/sex matched only) ended up 86% MR750 for
# controls vs 36% for patients, so a classifier could have scored ~75% by
# recognising the scanner rather than epilepsy. Here every patient is
# paired with a control FROM THE SAME SCANNER, same sex and same age band.
# EXCITE patients are unusable - there is no control to pair them with.
#
# Selection rules:
#   - PATIENTS: temporal-lobe cases only (Op_Type starts with "T"). IDEAS
#     also has frontal/parietal/occipital patients, which don't belong in a
#     TLE-vs-control study.
#   - Pairs are built per scanner: controls are chosen first (they are the
#     scarce group on every scanner), then each gets a patient from the
#     same scanner, same sex, same age band (falling back to the nearest
#     age band). Left/right surgery side is kept as balanced as possible.
#   - Ties are broken in favour of subjects already processed in this run,
#     then subjects already downloaded - this only saves compute/downloads,
#     it never overrides the scanner/sex/age match.
#   - Only subjects on their scanner's standard acquisition protocol
#     (STANDARD_PROTOCOL voxel sizes) are eligible - see the config note.
#   - Subjects that failed Step 1's registration QC are excluded.
#
# Age bands differ slightly between the two metadata files (patients use
# "15 to 20", "55 to 59", ...; controls use "Less than 20", "Over 55"),
# so both are mapped onto one shared scale before matching.
#
# Outputs (next to this script):
#   subject_list.txt         - one sub-ID per line, read by Steps 1-5
#   labels.tsv               - participant_id, group, pair, scanner, sex, age, TLE side
#   subjects_to_download.txt - selected subjects not yet on disk
#   scanner_by_subject.csv   - cache of scanner model per subject (from OpenNeuro JSONs)

import os
import glob
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from pipeline_config import RUN_DIR

# ============================================
# CONFIG
# ============================================

# Matched pairs per scanner (upper limits - capped at what is available).
# Only 11 HDx controls use the standard HDx protocol. On MR750 there are 67
# eligible controls but only ~60 eligible temporal patients, so MR750 is
# limited by PATIENTS: every eligible MR750 patient gets a matched control.
# (First cohort used 39 + 11 = 50 pairs; grown to ~71 pairs for more data.)
PAIRS_PER_SCANNER = {"DISCOVERY MR750": 60, "SIGNA HDx": 11}

# Eligible temporal patients that are NOT in a matched pair (mostly HDx,
# which has 100+ patients but only 11 controls) are added as a separate
# "lateralization_only" cohort. They go through Steps 1-11 and are used by
# Step 13 (left vs right TLE, a patients-only task) and by the label-free
# representation learning in Steps 8-11 - but NEVER by Step 12 (patient vs
# control), where they would bring back the scanner imbalance (e.g. 104 HDx
# patients vs 11 HDx controls) that the matching removed.
ADD_LATERALIZATION_PATIENTS = True

# Standard acquisition protocol per scanner, as (T1w, FLAIR) voxel sizes in
# mm from the BIDS JSON "VoxelSizes". Same scanner is not enough: 30 MR750
# patients were acquired with other protocols (e.g. 0.42 mm in-plane T1,
# 1.5 mm slices, 2 mm FLAIR slices) while every MR750 control used 1 mm
# isotropic T1 and FLAIR. The patch/wavelet features are resolution-
# sensitive, so an off-protocol patient would be identifiable by its
# texture alone. Only subjects on their scanner's standard protocol are
# eligible - the one used by (nearly) all of that scanner's controls.
STANDARD_PROTOCOL = {
    "DISCOVERY MR750": ("1.00x1.00x1.00", "1.00x1.00x1.00"),
    "SIGNA HDx": ("0.94x1.10x0.94", "0.94x0.94x5.00"),
}
RANDOM_SEED = 42

# Folder that contains the sub-1, sub-2, ... BIDS folders.
DATA_ROOT = r"C:\Users\pavne\OneDrive\Documents\Capstone"

PATIENT_METADATA = r"C:\Users\pavne\Downloads\25612080 (1)\Metadata_Release_Anon.csv"
CONTROL_METADATA = r"C:\Users\pavne\Downloads\Metadata_Controls_Release.csv"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SCANNER_CACHE = os.path.join(OUT_DIR, "scanner_by_subject.csv")
OPENNEURO_URL = "https://s3.amazonaws.com/openneuro.org/ds005602"

# Failed Step 1's registration MI gate (FLAIR alignment did not improve /
# fell short of -0.35), so never selected. 122/127/4053 failed in the first
# balanced run, 7/399/408/431 in the protocol-matched run (their rows were
# later lost from the Step 1 report, so they are listed here explicitly).
# Any new failures are also read from Step 1's report (registration_failures()).
REGISTRATION_FAILURES = {7, 122, 127, 399, 408, 431, 4053}

# A pair is only accepted if patient and control have the SAME SEX and age
# bands at most this far apart. Without this limit, using every MR750
# patient (patient-limited scanner) forced poor pairings - e.g. 5 opposite-
# sex pairs and controls skewing older - which would let sex/age leak into
# Step 12 as a shortcut. Patients without an acceptable partner go to the
# lateralization-only cohort instead.
MAX_AGE_BAND_GAP = 1

AGE_BAND_ORDER = ["<20", "20 to 24", "25 to 29", "30 to 34", "35 to 39",
                  "40 to 44", "45 to 49", "50 to 54", "55+"]


def harmonize_age_band(raw):
    raw = str(raw).strip()
    if raw.startswith(("Less than", "15 to")):
        return "<20"
    if raw.startswith(("Over", "55 to", "60 to", "65 to")):
        return "55+"
    if raw not in AGE_BAND_ORDER:
        raise ValueError(f"Unrecognised age band: {raw!r}")
    return raw


def subjects_on_disk(data_root):
    """Numeric IDs of subjects that have BOTH T1w and FLAIR on disk."""
    ids = set()
    for sub_dir in glob.glob(os.path.join(data_root, "sub-*")):
        sub_id = os.path.basename(sub_dir)
        anat = os.path.join(sub_dir, "anat")
        # size > 0: sub-176's T1w was on disk as a 0-byte failed download
        paths = [os.path.join(anat, f"{sub_id}_{m}.nii.gz") for m in ("T1w", "FLAIR")]
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths):
            ids.add(int(sub_id.split("-")[1]))
    return ids


def subjects_processed():
    """Numeric IDs already registered in this run (Step 1 output exists)."""
    reg_dir = os.path.join(RUN_DIR, "registered_outputs")
    return {int(os.path.basename(d).split("-")[1])
            for d in glob.glob(os.path.join(reg_dir, "sub-*"))}


def registration_failures():
    """Numeric IDs that failed Step 1's QC gate in this run: a row with
    passed=False in the Step 1 report and no registered output saved. The
    same -0.35 MI rule applies to everyone, so a failed subject is replaced
    by another matched one on the next Step 0 run."""
    report = os.path.join(RUN_DIR, "step1_registration_outputs", "step1_registration_report.csv")
    failed = set(REGISTRATION_FAILURES)
    if os.path.exists(report):
        rows = pd.read_csv(report)
        registered = subjects_processed()
        for sub_id in rows.loc[~rows["passed"].astype(bool), "sub_id"].unique():
            num = int(str(sub_id).split("-")[1])
            if num not in registered:
                failed.add(num)
    return failed


def _fetch_scanner(sub_num):
    """Scanner model and voxel sizes from the subject's JSON sidecars on
    OpenNeuro. A missing FLAIR sidecar means the subject has no FLAIR."""
    row = {"ID": sub_num}
    for modality in ("T1w", "FLAIR"):
        url = f"{OPENNEURO_URL}/sub-{sub_num}/anat/sub-{sub_num}_{modality}.json"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                meta = json.load(resp)
            row[f"{modality}_scanner"] = meta.get("ManufacturersModelName")
            row[f"{modality}_vox"] = "x".join(f"{v:.2f}" for v in meta.get("VoxelSizes", []))
        except Exception:
            row[f"{modality}_scanner"] = None
            row[f"{modality}_vox"] = None
    return row


def load_scanners(ids):
    """Scanner + voxel sizes per subject, fetched once and cached in
    SCANNER_CACHE (re-fetched if the cache predates the voxel columns)."""
    cache = pd.read_csv(SCANNER_CACHE) if os.path.exists(SCANNER_CACHE) else pd.DataFrame(columns=["ID"])
    if "T1w_vox" not in cache.columns:
        cache = pd.DataFrame(columns=["ID"])
    missing = sorted(set(ids) - set(cache["ID"]))
    if missing:
        print(f"Fetching scanner info for {len(missing)} subject(s) from OpenNeuro...")
        with ThreadPoolExecutor(16) as ex:
            fetched = pd.DataFrame(list(ex.map(_fetch_scanner, missing)))
        cache = pd.concat([cache, fetched], ignore_index=True)
        cache.to_csv(SCANNER_CACHE, index=False)
    return cache.set_index("ID")


def priority(ids, processed, on_disk):
    """0 = already processed, 1 = downloaded only, 2 = needs downloading."""
    return np.where(ids.isin(processed), 0, np.where(ids.isin(on_disk), 1, 2))


def build_pairs(controls, patients, scanner, n_pairs, side_counts, rng):
    """Build up to n_pairs (control, patient) pairs on one scanner.

    The SCARCER group on this scanner is the "anchor": anchors are picked
    first (already-processed subjects preferred, so earlier work is reused),
    then each anchor gets the best partner from the other group - same sex
    and age band, else same sex and nearest band, else nearest band. On HDx
    controls are scarce (11); on MR750 patients are (~60 vs 67 controls).
    When the partner is a patient, the currently under-represented surgery
    side is preferred so left/right stays balanced."""
    ctrl = controls[controls["scanner"] == scanner].copy()
    pat = patients[patients["scanner"] == scanner].copy()
    available = min(len(ctrl), len(pat))
    if available < n_pairs:
        print(f"{scanner}: {n_pairs} pairs requested, only {available} possible "
              f"({len(ctrl)} controls, {len(pat)} patients) - using {available}")
        n_pairs = available

    anchors_are_patients = len(pat) <= len(ctrl)
    anchors, pool = (pat, ctrl) if anchors_are_patients else (ctrl, pat)
    anchors = anchors.assign(rand=rng.random(len(anchors))).sort_values(["priority", "rand"]).head(n_pairs)
    pool = pool.assign(rand=rng.random(len(pool)), used=False)

    pairs = []
    # match the hardest-to-match anchors (rarest sex/age band in the pool)
    # first, so they get first pick
    rarity = anchors.apply(lambda a: ((pool["Sex"] == a["Sex"]) & (pool["age_idx"] == a["age_idx"])).sum(), axis=1)
    for _, a in anchors.assign(rarity=rarity).sort_values("rarity").iterrows():
        free = pool[~pool["used"]]
        dist = (free["age_idx"] - a["age_idx"]).abs()
        same_sex = free["Sex"] == a["Sex"]
        if (same_sex & (dist == 0)).any():
            cand, quality = free[same_sex & (dist == 0)], "exact"
        elif (same_sex & (dist <= MAX_AGE_BAND_GAP)).any():
            cand = free[same_sex & (dist == dist[same_sex].min())]
            quality = "sex, nearest age"
        else:
            continue  # no acceptable partner - this anchor stays unpaired

        if anchors_are_patients:
            chosen = cand.sort_values(["priority", "rand"]).iloc[0]
            control, patient = chosen, a
        else:
            rare_side = min(side_counts, key=side_counts.get)
            cand = cand.assign(side_rank=(cand["Op_Side"] != rare_side).astype(int))
            chosen = cand.sort_values(["side_rank", "priority", "rand"]).iloc[0]
            control, patient = a, chosen

        pool.loc[chosen.name, "used"] = True
        side_counts[patient["Op_Side"]] = side_counts.get(patient["Op_Side"], 0) + 1
        pairs.append((control, patient, quality))
    return pairs


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    on_disk = subjects_on_disk(DATA_ROOT)
    processed = subjects_processed()

    patients = pd.read_csv(PATIENT_METADATA).drop_duplicates("ID")  # one ID appears twice
    patients = patients[patients["Op_Type"].str.startswith("T")].copy()
    controls = pd.read_csv(CONTROL_METADATA)

    scanners = load_scanners(list(patients["ID"]) + list(controls["ID"]))
    for df in (patients, controls):
        df["scanner"] = df["ID"].map(scanners["T1w_scanner"])
        df["has_flair"] = df["ID"].map(scanners["FLAIR_scanner"]).notna()
        protocol = list(zip(df["ID"].map(scanners["T1w_vox"]), df["ID"].map(scanners["FLAIR_vox"])))
        df["standard_protocol"] = [p == STANDARD_PROTOCOL.get(s) for p, s in zip(protocol, df["scanner"])]
        df["age_band"] = df["Binned_Age_at_Scan"].map(harmonize_age_band)
        df["age_idx"] = df["age_band"].map(AGE_BAND_ORDER.index)
        df["priority"] = priority(df["ID"], processed, on_disk)

    failed = registration_failures()
    print(f"Excluded for failing registration QC: {sorted(failed)}")
    patients = patients[patients["has_flair"] & patients["standard_protocol"]
                        & ~patients["ID"].isin(failed)]
    controls = controls[controls["has_flair"] & controls["standard_protocol"]
                        & ~controls["ID"].isin(failed)]
    print("Eligible subjects by scanner (T1w+FLAIR, standard protocol, passed registration):")
    print(pd.concat([controls.assign(group="control"), patients.assign(group="patient")])
          .pipe(lambda d: pd.crosstab(d["scanner"], d["group"])))

    side_counts = {"L": 0, "R": 0}
    rows, pair_id = [], 0
    for scanner, n_pairs in PAIRS_PER_SCANNER.items():
        for c, p, quality in build_pairs(controls, patients, scanner, n_pairs, side_counts, rng):
            pair_id += 1
            for subj, group in ((p, "patient"), (c, "control")):
                rows.append({
                    "participant_id": f"sub-{int(subj['ID'])}",
                    "group": group,
                    "cohort": "matched",
                    "pair": pair_id,
                    "scanner": scanner,
                    "sex": subj["Sex"],
                    "age_band": subj["age_band"],
                    "tle_side": subj["Op_Side"] if group == "patient" else "n/a",
                    "pathology": subj["Pathology"] if group == "patient" else "n/a",
                    "match_quality": quality,
                })

    if ADD_LATERALIZATION_PATIENTS:
        paired = {r["participant_id"] for r in rows}
        extra = patients[patients["scanner"].isin(PAIRS_PER_SCANNER)
                         & ~("sub-" + patients["ID"].astype(str)).isin(paired)]
        for _, subj in extra.sort_values("ID").iterrows():
            rows.append({
                "participant_id": f"sub-{int(subj['ID'])}",
                "group": "patient",
                "cohort": "lateralization_only",
                "pair": "",
                "scanner": subj["scanner"],
                "sex": subj["Sex"],
                "age_band": subj["age_band"],
                "tle_side": subj["Op_Side"],
                "pathology": subj["Pathology"],
                "match_quality": "",
            })
    labels = pd.DataFrame(rows)

    # Sanity check on the ID rule the rest of the pipeline relies on.
    ids = labels["participant_id"].str.split("-").str[1].astype(int)
    assert ((ids >= 4001) == (labels["group"] == "control")).all(), "ID rule violated"
    assert labels["participant_id"].is_unique
    assert (labels.loc[labels["cohort"] != "matched", "group"] == "patient").all()

    labels.to_csv(os.path.join(OUT_DIR, "labels.tsv"), sep="\t", index=False)
    with open(os.path.join(OUT_DIR, "subject_list.txt"), "w") as f:
        f.write("\n".join(labels["participant_id"]) + "\n")

    to_download = sorted(labels.loc[~ids.isin(on_disk), "participant_id"],
                         key=lambda s: int(s.split("-")[1]))
    with open(os.path.join(OUT_DIR, "subjects_to_download.txt"), "w") as f:
        f.write("\n".join(to_download) + ("\n" if to_download else ""))

    # ---- report ----
    matched = labels[labels["cohort"] == "matched"]
    pats = matched[matched["group"] == "patient"]
    all_pats = labels[labels["group"] == "patient"]
    print(f"\nMATCHED cohort (Steps 1-13, incl. Step 12): {len(pats)} patients + {len(matched) - len(pats)} controls")
    print("\nScanner by group:")
    print(pd.crosstab(matched["scanner"], matched["group"]))
    print("\nSex by group:")
    print(pd.crosstab(matched["group"], matched["sex"]))
    print("\nAge band by group:")
    pd.set_option("display.width", 200)
    print(pd.crosstab(matched["group"], matched["age_band"])[
        [b for b in AGE_BAND_ORDER if b in set(matched["age_band"])]])
    print("\nPair match quality:", pats["match_quality"].value_counts().to_dict())
    print("Matched patients TLE side:", pats["tle_side"].value_counts().to_dict())
    print(f"\nLATERALIZATION-ONLY patients (Steps 1-11 + 13, never Step 12): "
          f"{int((labels['cohort'] == 'lateralization_only').sum())}")
    print("ALL patients for Step 13, side by scanner:")
    print(pd.crosstab(all_pats["scanner"], all_pats["tle_side"], margins=True))
    print("All patients pathology:", all_pats["pathology"].value_counts().to_dict())
    n_new = int((~ids.isin(processed)).sum())
    print(f"\n{n_new} selected subject(s) not yet processed by Step 1; "
          f"{len(to_download)} need downloading -> subjects_to_download.txt")
    print(f"Wrote subject_list.txt and labels.tsv -> {OUT_DIR}")
