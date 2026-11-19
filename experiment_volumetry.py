# ============================================
# BASELINE: HIPPOCAMPAL / AMYGDALA VOLUME (FreeSurfer, from IDEAS)
# ============================================
# The classic clinical marker of hippocampal sclerosis. Uses the volume
# tables IDEAS publishes on figshare (FreeSurfer aseg: patients + controls),
# no image processing. Development subjects only; same CV as the other
# experiments (tle: matched pairs together, 5-fold x 10; side: stratified).
#
# Features (volumes divided by intracranial volume, eTIV):
#   tle  : hippocampus L, R, amygdala L, R, and |asymmetry| of both
#   side : asymmetry index (L - R) / (L + R) of hippocampus and amygdala
#          (the side of the smaller hippocampus = the side of HS)
#
# Run:  python experiment_volumetry.py <folder with aseg_vol.txt and controls_aseg_vol.txt>

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_config import LABELS_FILE, RUN_DIR, dev_only

N_FOLDS, N_REPEATS, SEED = 5, 10, 42
ROWS = ["Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala", "EstimatedTotalIntraCranialVol"]


def load_volumes(folder):
    tabs = []
    for f in ("aseg_vol.txt", "controls_aseg_vol.txt"):
        t = pd.read_csv(os.path.join(folder, f), sep=None, engine="python").set_index("Measure:volume")
        tabs.append(t.loc[ROWS].T)
    v = pd.concat(tabs).astype(float)
    v.index = ["sub-" + str(i).strip() for i in v.index]
    v = v[~v.index.duplicated()]
    icv = v["EstimatedTotalIntraCranialVol"]
    hl, hr = v["Left-Hippocampus"] / icv * 1e3, v["Right-Hippocampus"] / icv * 1e3
    al, ar = v["Left-Amygdala"] / icv * 1e3, v["Right-Amygdala"] / icv * 1e3
    return pd.DataFrame({"hipp_L": hl, "hipp_R": hr, "amyg_L": al, "amyg_R": ar,
                         "hipp_AI": (hl - hr) / (hl + hr), "amyg_AI": (al - ar) / (al + ar)})


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else np.nan
    sens, spec, ppv = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    return {"accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
            "f1": div(2 * ppv * sens, ppv + sens), "auc": roc_auc_score(y, prob), "mcc": matthews_corrcoef(y, pred)}


def cv(X, y, groups):
    rows = []
    for r in range(N_REPEATS):
        prob = np.zeros(len(y))
        if groups is not None:
            pairs = np.random.default_rng(SEED + r).permutation(np.unique(groups))
            folds = [(np.where(~np.isin(groups, p))[0], np.where(np.isin(groups, p))[0]) for p in np.array_split(pairs, N_FOLDS)]
        else:
            folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(X, y)
        for tr, te in folds:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[tr], y[tr])
            prob[te] = m.predict_proba(X[te])[:, 1]
        rows.append(metrics(y, prob))
    return pd.DataFrame(rows).agg(["mean", "std"]).T


if __name__ == "__main__":
    vol = load_volumes(sys.argv[1])
    lab = dev_only(pd.read_csv(LABELS_FILE, sep="\t")).set_index("participant_id")
    out_dir = os.path.join(RUN_DIR, "experiments")
    os.makedirs(out_dir, exist_ok=True)

    m = lab[lab["cohort"] == "matched"]
    m = m[m.index.isin(vol.index)]
    X = np.column_stack([vol.loc[m.index, ["hipp_L", "hipp_R", "amyg_L", "amyg_R"]].to_numpy(),
                         vol.loc[m.index, ["hipp_AI", "amyg_AI"]].abs().to_numpy()])
    y = (m["group"] == "patient").astype(int).to_numpy()
    res_tle = cv(X, y, m["pair"].to_numpy())
    print(f"TLE vs control, {len(m)} development subjects (volumes found for all: {len(m) == (lab['cohort'] == 'matched').sum()})")
    print(res_tle.round(3).to_string())

    p = lab[(lab["group"] == "patient") & lab.index.isin(vol.index)]
    Xs = vol.loc[p.index, ["hipp_AI", "amyg_AI"]].to_numpy()
    ys = (p["tle_side"] == "L").astype(int).to_numpy()
    res_side = cv(Xs, ys, None)
    print(f"\nLeft vs right, {len(p)} development patients")
    print(res_side.round(3).to_string())
    hs = p["pathology"].eq("HS").to_numpy()
    rule = (vol.loc[p.index, "hipp_AI"].to_numpy() < 0).astype(int)        # smaller left hippocampus -> left TLE
    print(f"\nNo-training rule 'smaller hippocampus = epileptic side': accuracy {np.mean(rule == ys):.3f} "
          f"(HS {np.mean(rule[hs] == ys[hs]):.3f}, non-HS {np.mean(rule[~hs] == ys[~hs]):.3f})")
    pd.concat({"tle": res_tle, "side": res_side}).to_csv(os.path.join(out_dir, "volumetry_baseline.csv"))
