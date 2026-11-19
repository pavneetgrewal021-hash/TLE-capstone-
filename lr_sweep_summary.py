# ============================================
# LEARNING-RATE SENSITIVITY OF THE FINAL CNNs (development CV only)
# ============================================
# gpu_hippo_cnn.py was run with TLE_CNN_LR = each value below, for both
# boxes and both tasks, on the same folds as the locked models (lr 1e-3).
# This reports, per learning rate (mean +- std over the 5 repeats):
#   hippo_cnn, temporal_cnn   each CNN alone
#   final                     the locked ensemble with that learning rate
#                             (tle: temporal + tabular; side: hippo + temporal + tabular)
# A sensitivity check, not model selection: the locked models stay as they
# are and the held-out test is not rerun.
#
# Run:  python lr_sweep_summary.py tle     (or side)

import os
import sys

import numpy as np
import pandas as pd

from experiment_classifiers import metrics
from pipeline_config import RUN_DIR

LRS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]
DEFAULT = "1e-3"                                                            # the locked models' value
FINAL = {"tle": ["temporal_cnn", "tabular"], "side": ["hippo_cnn", "temporal_cnn", "tabular"]}


def oof(crop, task, lr):
    folder = task if lr == DEFAULT else f"{task}_lr{lr}"
    path = os.path.join(RUN_DIR, "gpu_hippo", crop, folder, "oof_predictions.csv")
    if lr == DEFAULT and not os.path.exists(path) and crop == "hippo_crops":
        path = os.path.join(RUN_DIR, "gpu_hippo", folder, "oof_predictions.csv")   # first run wrote one level higher
    return pd.read_csv(path) if os.path.exists(path) else None


def main(task):
    rows = []
    for lr in LRS:
        h, t = oof("hippo_crops", task, lr), oof("temporal_crops", task, lr)
        if h is None or t is None:
            print(f"lr {lr}: missing run, skipped")
            continue
        df = h.rename(columns={"cnn": "hippo_cnn"}).merge(
            t[["repeat", "subject", "label", "cnn"]].rename(columns={"cnn": "temporal_cnn"}),
            on=["repeat", "subject", "label"], validate="1:1")
        for name, cols in (("hippo_cnn", ["hippo_cnn"]), ("temporal_cnn", ["temporal_cnn"]), ("final", FINAL[task])):
            per = pd.DataFrame([metrics(g["label"].to_numpy(), g[cols].mean(axis=1).to_numpy())
                                for _, g in df.groupby("repeat")])
            rows.append({"lr": lr, "model": name, "accuracy": per["accuracy"].mean(), "acc_std": per["accuracy"].std(),
                         "auc": per["auc"].mean(), "auc_std": per["auc"].std(), "f1": per["f1"].mean(),
                         "mcc": per["mcc"].mean()})
    out = pd.DataFrame(rows)
    path = os.path.join(RUN_DIR, "gpu_hippo", f"lr_sweep_{task}.csv")
    out.to_csv(path, index=False)
    print(f"{task}: learning-rate sensitivity, development CV (5-fold x 5 repeats); locked models use lr {DEFAULT}")
    for name in ("final", "temporal_cnn", "hippo_cnn"):
        print(f"\n{name}")
        print(out[out["model"] == name].drop(columns="model").round(3).to_string(index=False))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "tle"
    if task not in ("tle", "side"):
        sys.exit("task must be 'tle' or 'side'")
    main(task)
