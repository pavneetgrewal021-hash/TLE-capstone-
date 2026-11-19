# ============================================
# ENSEMBLES FROM SAVED OUT-OF-FOLD PREDICTIONS
# ============================================
# gpu_hippo_cnn.py saves, for every repeat, each development subject's
# predicted probability from a model that never saw that subject. All runs
# use identical folds, so averaging those probabilities (fixed equal
# weights, nothing fitted) is a fair ensemble.
#
# Compares, per task: hippocampus CNN, temporal-lobe CNN, tabular model
# (volumes [+ PCA features]) and every average of them; mean over repeats.
# Selection rule fixed in advance: highest mean AUC.
#
# Run:  python combine_oof.py tle      (or side)

import itertools
import os
import sys

import numpy as np
import pandas as pd

from experiment_classifiers import metrics
from pipeline_config import RUN_DIR


def load(task):
    base = os.path.join(RUN_DIR, "gpu_hippo")
    hippo = os.path.join(base, "hippo_crops", task, "oof_predictions.csv")
    if not os.path.exists(hippo):                                       # first run wrote one level higher
        hippo = os.path.join(base, task, "oof_predictions.csv")
    h = pd.read_csv(hippo).rename(columns={"cnn": "hippo_cnn"})
    t = pd.read_csv(os.path.join(base, "temporal_crops", task, "oof_predictions.csv")).rename(columns={"cnn": "temporal_cnn"})
    m = h.merge(t[["repeat", "subject", "label", "temporal_cnn"]], on=["repeat", "subject", "label"], validate="1:1")
    assert len(m) == len(h) == len(t), "runs do not cover the same subjects/repeats"
    return m[["repeat", "subject", "label", "hippo_cnn", "temporal_cnn", "tabular"]]


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "tle"
    df = load(task)
    models = ["hippo_cnn", "temporal_cnn", "tabular"]
    rows = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(models, k):
            per = [metrics(g["label"].to_numpy(), g[list(combo)].mean(axis=1).to_numpy())
                   for _, g in df.groupby("repeat")]
            per = pd.DataFrame(per)
            rows.append({"model": " + ".join(combo), **per.mean().to_dict(),
                         "auc_std": per["auc"].std(), "acc_std": per["accuracy"].std()})
    out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    cols = ["model", "accuracy", "acc_std", "auc", "auc_std", "sensitivity", "specificity", "f1", "mcc"]
    print(f"{task}: {df['subject'].nunique()} subjects, {df['repeat'].nunique()} repeats (mean over repeats)")
    print(out[cols].round(3).to_string(index=False))
    out.to_csv(os.path.join(RUN_DIR, "gpu_hippo", f"ensembles_{task}.csv"), index=False)
    print(f"\nselected by the fixed rule (highest AUC): {out.iloc[0]['model']}")
