# ============================================
# RUN PER-SUBJECT STEPS IN PARALLEL (GPU server)
# ============================================
# Runs Steps 1-4 as N parallel shards (each process takes every N-th
# subject via TLE_SHARD), waits for all shards of a step before starting
# the next, then merges the per-shard report CSVs. Steps 5-13 work on the
# whole cohort at once and are run normally afterwards.
#
# Run:  python run_parallel.py [n_shards]        (default 24)
# Logs: RUN_DIR/logs/parallel/<step>_shard<i>.log

import glob
import os
import subprocess
import sys
import time

import pandas as pd

from pipeline_config import CODE_DIR, RUN_DIR

# (script, report, max shards). Step 2 runs HD-BET on the GPU when
# TLE_HDBET_DEVICE=cuda; each HD-BET process loads its own network onto the
# 40 GB GPU slice, so that step uses fewer parallel shards (on the server HD-BET runs on CPU, TLE_HDBET_DEVICE=cpu: GPU mode was not faster).
STEPS = [
    ("step1_registration.py", "step1_registration_outputs/step1_registration_report.csv", None),
    ("step2_skull_stripping.py", "skullstripped_outputs/step2_skullstrip_report.csv", 48),
    ("step3_bias_correction.py", "biascorrected_outputs/step3_biascorrection_report.csv", None),
    ("step4_intensity_normalization.py", "normalized_outputs/step4_normalization_report.csv", None),
]
# Each shard is one process; cap its threads so N shards share the CPUs
# instead of every SimpleITK/numpy call grabbing all 224.
THREADS_PER_SHARD = 4


def run_step(script, n_shards, log_dir):
    env = dict(os.environ)
    for var in ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = str(THREADS_PER_SHARD)
    procs = []
    for i in range(n_shards):
        log = open(os.path.join(log_dir, f"{script[:-3]}_shard{i}.log"), "w")
        procs.append((subprocess.Popen(
            [sys.executable, "-u", os.path.join(CODE_DIR, script)],
            env={**env, "TLE_SHARD": f"{i}/{n_shards}"}, stdout=log, stderr=subprocess.STDOUT), log))
    start = time.time()
    while any(p.poll() is None for p, _ in procs):
        done = sum(p.poll() is not None for p, _ in procs)
        print(f"  {script}: {done}/{n_shards} shards finished ({time.time() - start:.0f} s)", flush=True)
        time.sleep(30)
    for p, log in procs:
        log.close()
    failed = [i for i, (p, _) in enumerate(procs) if p.returncode != 0]
    return failed


def merge_reports(report_rel):
    root, ext = os.path.splitext(os.path.join(RUN_DIR, report_rel))
    parts = sorted(glob.glob(f"{root}_shard*{ext}"))
    if not parts:
        return 0
    frames = []
    for p in parts:
        try:
            frames.append(pd.read_csv(p))
        except pd.errors.EmptyDataError:  # a shard whose subjects were all skipped writes an empty report
            pass
    if not frames:
        return 0
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=[c for c in ("sub_id", "direction", "modality") if c in merged.columns], keep="last")
    merged.to_csv(root + ext, index=False)
    return len(merged)


if __name__ == "__main__":
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    log_dir = os.path.join(RUN_DIR, "logs", "parallel")
    os.makedirs(log_dir, exist_ok=True)
    for script, report, max_shards in STEPS:
        n = min(n_shards, max_shards) if max_shards else n_shards
        print(f"\n########## {script} x {n} shards ##########", flush=True)
        failed = run_step(script, n, log_dir)
        rows = merge_reports(report)
        print(f"  merged report: {rows} rows -> {report}", flush=True)
        if failed:
            sys.exit(f"{script}: shard(s) {failed} exited with an error - see {log_dir}")
    print("\nSteps 1-4 complete. Next: step5_patch_extraction.py ... step13_lateralization.py")
