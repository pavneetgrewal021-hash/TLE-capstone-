# ============================================
# SHARED PIPELINE CONFIG
# ============================================
# Every step imports its paths from here, so the scripts can be run from
# any folder and all outputs of one run land together under RUN_DIR.
# To start a completely separate run, change RUN_DIR (and re-run Step 0
# if the cohort should change).

import os

# Folder containing the sub-1, sub-4001, ... BIDS subject folders.
# TLE_DATA_ROOT / TLE_RUN_DIR override these, e.g. on the GPU server.
DATA_ROOT = os.environ.get("TLE_DATA_ROOT", r"C:\Users\pavne\OneDrive\Documents\Capstone")

# Where every step writes its outputs for this run.
RUN_DIR = os.environ.get("TLE_RUN_DIR", os.path.join(DATA_ROOT, "run_balanced"))

# Optional "i/n": this process handles only every n-th subject (offset i),
# so n copies of a per-subject step can run in parallel (run_parallel.py).
SHARD = os.environ.get("TLE_SHARD")

# Written by step0_select_subjects.py (same folder as this file).
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT_LIST_FILE = os.path.join(CODE_DIR, "subject_list.txt")
LABELS_FILE = os.path.join(CODE_DIR, "labels.tsv")

os.makedirs(RUN_DIR, exist_ok=True)


def load_subject_list():
    """The fixed cohort chosen by Step 0, as a set of sub-IDs."""
    if not os.path.exists(SUBJECT_LIST_FILE):
        raise FileNotFoundError(
            f"{SUBJECT_LIST_FILE} not found - run step0_select_subjects.py first."
        )
    with open(SUBJECT_LIST_FILE) as f:
        subjects = sorted(line.strip() for line in f if line.strip())
    if SHARD:
        i, n = (int(x) for x in SHARD.split("/"))
        subjects = subjects[i::n]
    return set(subjects)


def dev_only(labels_df):
    """Steps 12/13 and all experiments see only the development subjects.
    The held-out test subjects (make_heldout_split.py) are touched once, at
    the very end, by a separate final-evaluation script (train on dev,
    predict test)."""
    if "split" not in labels_df.columns:
        return labels_df
    return labels_df[labels_df["split"] == "dev"]


def report_file(path):
    """Per-shard report name, so parallel shards don't overwrite each
    other's report CSV (merge them afterwards)."""
    if not SHARD:
        return path
    i, n = SHARD.split("/")
    root, ext = os.path.splitext(path)
    return f"{root}_shard{i}of{n}{ext}"
