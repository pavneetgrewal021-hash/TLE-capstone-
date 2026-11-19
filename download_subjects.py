# ============================================
# DOWNLOAD SELECTED SUBJECTS FROM OPENNEURO
# ============================================
# Reads subjects_to_download.txt (written by step0_select_subjects.py) and
# fetches each subject's T1w + FLAIR (+ JSON sidecars) from OpenNeuro's
# public S3 bucket into DATA_ROOT, in the same BIDS layout as the patients
# already there (DATA_ROOT/sub-4001/anat/sub-4001_T1w.nii.gz, ...).
# Works for patients and controls alike.
# ~9 MB per subject. Safe to re-run: files already on disk are skipped.

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from pipeline_config import DATA_ROOT
LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subjects_to_download.txt")
BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds005602"
SUFFIXES = ["T1w.nii.gz", "T1w.json", "FLAIR.nii.gz", "FLAIR.json"]


def replace_with_retry(src, dest, attempts=10):
    """DATA_ROOT is inside OneDrive, which briefly locks every new file
    to sync it - a rename in that window fails with WinError 32. Wait
    and retry instead of giving up."""
    for i in range(attempts):
        try:
            os.replace(src, dest)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(1 + i)

# Several downloads run at once: on a throttled network each connection is
# slow (~100 KB/s on campus Wi-Fi) but 6 in parallel reached ~850 KB/s.
N_PARALLEL = 8
# A connection that delivers nothing for this long is abandoned and retried
# (some connections stall almost completely, e.g. ~200 B/s).
STALL_TIMEOUT_S = 30
MAX_ATTEMPTS = 6


def fetch(url, dest):
    """Download url -> dest via a .part file, checking the byte count
    against Content-Length so a cut-off transfer is never accepted."""
    part = dest + ".part"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=STALL_TIMEOUT_S) as resp, open(part, "wb") as out:
                expected = int(resp.headers.get("Content-Length", -1))
                received = 0
                while chunk := resp.read(256 * 1024):
                    out.write(chunk)
                    received += len(chunk)
            if expected >= 0 and received != expected:
                raise IOError(f"got {received} of {expected} bytes")
            replace_with_retry(part, dest)
            return
        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2 * attempt)


def download_subject(sub_id):
    anat_dir = os.path.join(DATA_ROOT, sub_id, "anat")
    os.makedirs(anat_dir, exist_ok=True)
    errors = []
    for suffix in SUFFIXES:
        name = f"{sub_id}_{suffix}"
        dest = os.path.join(anat_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        try:
            fetch(f"{BASE_URL}/{sub_id}/anat/{name}", dest)
        except Exception as e:
            errors.append(f"{name}: {e}")
    return sub_id, errors


if __name__ == "__main__":
    # Optional: pass sub-IDs to (re)fetch specific subjects instead of the
    # list file, e.g. to repair a truncated file:  download_subjects.py sub-176
    if len(sys.argv) > 1:
        subjects = sys.argv[1:]
    else:
        with open(LIST_FILE) as f:
            subjects = [line.strip() for line in f if line.strip()]

    failed = []
    with ThreadPoolExecutor(N_PARALLEL) as pool:
        for i, (sub_id, errors) in enumerate(pool.map(download_subject, subjects), start=1):
            failed.extend(errors)
            print(f"[{i}/{len(subjects)}] {sub_id} {'done' if not errors else 'FAILED'}", flush=True)

    if failed:
        print(f"\n{len(failed)} file(s) failed - re-run this script to retry:")
        print("\n".join(failed))
    else:
        print(f"\nAll {len(subjects)} subjects downloaded to {DATA_ROOT}")
