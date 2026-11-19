# ============================================
# HELD-OUT TEST SET (made once, 2026-10-25, before the GPU experiments)
# ============================================
# Locks away ~20% of the cohort that no experiment may look at until the
# final pipeline is chosen. It is then evaluated ONCE (TLE_EVAL_TEST=1).
#
#   Step 12 (patient vs control): 12 of the 59 matched PAIRS, both members,
#     drawn per scanner in proportion (10 MR750 + 2 HDx), with left/right
#     TLE balanced among the held-out patients.
#   Step 13 (left vs right): those 12 patients + 20 of the 101
#     lateralization-only patients, stratified by scanner and side.
#
# Writes heldout_test.txt and a `split` column (dev / test) in labels.tsv.
# Refuses to run again if the split already exists, so it can't be redrawn.

import os
import sys

import numpy as np
import pandas as pd

from pipeline_config import CODE_DIR, LABELS_FILE

SEED = 20261025
TEST_PAIRS = {"DISCOVERY MR750": 10, "SIGNA HDx": 2}
TEST_LAT_ONLY = 20
OUT_FILE = os.path.join(CODE_DIR, "heldout_test.txt")

if __name__ == "__main__":
    labels = pd.read_csv(LABELS_FILE, sep="\t")
    if "split" in labels.columns or os.path.exists(OUT_FILE):
        sys.exit("Held-out split already exists - not redrawing it.")
    rng = np.random.default_rng(SEED)
    test = set()

    matched = labels[labels["cohort"] == "matched"]
    pat = matched[matched["group"] == "patient"]
    for scanner, n_pairs in TEST_PAIRS.items():
        pool = pat[pat["scanner"] == scanner]
        # half of this scanner's held-out pairs from left-TLE patients, half right
        n_left = n_pairs // 2
        for side, n in (("L", n_left), ("R", n_pairs - n_left)):
            cand = pool[pool["tle_side"] == side]["pair"].to_numpy()
            for p in rng.choice(cand, size=n, replace=False):
                test.update(matched[matched["pair"] == p]["participant_id"])

    lat = labels[labels["cohort"] == "lateralization_only"]
    groups = lat.groupby(["scanner", "tle_side"])
    for _, g in groups:
        n = int(round(TEST_LAT_ONLY * len(g) / len(lat)))
        test.update(rng.choice(g["participant_id"].to_numpy(), size=n, replace=False))

    labels["split"] = np.where(labels["participant_id"].isin(test), "test", "dev")
    labels.to_csv(LABELS_FILE, sep="\t", index=False)
    with open(OUT_FILE, "w", newline="\n") as f:
        f.write("\n".join(sorted(test)) + "\n")

    t = labels[labels["split"] == "test"]
    print(f"Held out {len(t)} subjects -> {OUT_FILE}")
    print(t.groupby(["cohort", "group", "scanner"]).size().to_string())
    print("held-out patients by side:", t[t["group"] == "patient"]["tle_side"].value_counts().to_dict())
