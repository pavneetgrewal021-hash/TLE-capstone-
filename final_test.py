# ============================================
# FINAL HELD-OUT TEST (run ONCE)
# ============================================
# Models locked from the development CV (combine_oof.py, rule fixed in
# advance: highest mean AUC):
#   tle  : temporal-lobe CNN + tabular (LDA on volumes + PCA avg/|L-R|)
#   side : hippocampus CNN + temporal-lobe CNN + tabular (LR on volume asymmetry)
# Final probability = plain average of the members (no weights fitted).
#
# Every member is trained on ALL development subjects (tle: 47 pairs,
# side: 128 patients) and predicts the 44 held-out subjects exactly once.
# Each CNN is trained with 5 seeds and the 5 probabilities are averaged, so
# the result does not hang on one lucky / unlucky initialisation.
#
# Reports: all metrics, 95% bootstrap CIs (tle: resampling matched pairs,
# side: resampling patients), binomial p-value of the accuracy vs 50%,
# HS / non-HS accuracy, and every member alone (for transparency; the
# ensemble above is the pre-selected primary result).
#
# Output: RUN_DIR/final_test/<task>/  (refuses to run again once it exists)
# Run:    TLE_VOL_DIR=fs_tables python final_test.py tle    (then: side)
#         TLE_MIL_SMOKE=1 ...  -> tiny CPU check on whatever crops exist

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_classifiers import load_features, metrics
from experiment_volumetry import load_volumes
from gpu_hippo_cnn import predict_cnn, train_cnn
from pipeline_config import LABELS_FILE, RUN_DIR

SMOKE = os.environ.get("TLE_MIL_SMOKE") == "1"
N_SEEDS = 1 if SMOKE else int(os.environ.get("TLE_FINAL_SEEDS", "5"))
N_BOOT, SEED = 2000, 42
MEMBERS = {"tle": ["temporal_cnn", "tabular"],
           "side": ["hippo_cnn", "temporal_cnn", "tabular"]}
CROPS = {"hippo_cnn": "hippo_crops", "temporal_cnn": "temporal_crops"}


def select(lab, task):
    if task == "tle":
        return lab[lab["cohort"] == "matched"].sort_values("participant_id")
    return lab[lab["group"] == "patient"].sort_values("participant_id")


def target(df, task):
    return ((df["group"] == "patient") if task == "tle" else (df["tle_side"] == "L")).astype(int).to_numpy()


def tabular(task, train, test, dev_controls):
    vol = load_volumes(os.environ.get("TLE_VOL_DIR", "fs_tables"))
    if task == "tle":
        feats, _ = load_features(train + test, dev_controls)
        v = vol.loc[train + test]
        T = np.hstack([v[["hipp_L", "hipp_R", "amyg_L", "amyg_R"]].to_numpy(),
                       v[["hipp_AI", "amyg_AI"]].abs().to_numpy(), feats["pca_avg_absdiff"]])
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    else:
        T = vol.loc[train + test][["hipp_AI", "amyg_AI"]].to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return T[:len(train)], T[len(train):], model


def bootstrap(y, p, clusters, rng):
    ids = np.unique(clusters)
    acc, auc = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        acc.append(np.mean((p[idx] >= 0.5) == y[idx]))
        auc.append(roc_auc_score(y[idx], p[idx]))
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5)) if a else (np.nan, np.nan)
    return ci(acc), ci(auc)


def main(task):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(RUN_DIR, "final_test", task + ("_smoke" if SMOKE else ""))
    if not SMOKE and os.path.exists(os.path.join(out_dir, "final_metrics.csv")):
        sys.exit(f"{out_dir} already has results - the held-out test is run only once.")

    lab = pd.read_csv(LABELS_FILE, sep="\t")
    dev_controls = sorted(lab[(lab["split"] == "dev") & (lab["group"] == "control")]["participant_id"])
    dev, test = select(lab[lab["split"] == "dev"], task), select(lab[lab["split"] == "test"], task)
    if SMOKE:                                                           # laptop has crops for a few subjects only
        have = lambda s: all(os.path.exists(os.path.join(RUN_DIR, c, f"{s}.npy")) for c in CROPS.values())
        dev, test = dev[dev["participant_id"].map(have)], test[test["participant_id"].map(have)]
    train, tst = dev["participant_id"].tolist(), test["participant_id"].tolist()
    y_tr, y_te = target(dev, task), target(test, task)
    print(f"task={task} device={device} train={len(train)} test={len(tst)} members={MEMBERS[task]} "
          f"cnn_seeds={N_SEEDS}", flush=True)

    probs = {}
    for m in MEMBERS[task]:
        if m == "tabular":
            Ttr, Tte, model = tabular(task, train, tst, dev_controls)
            probs[m] = model.fit(Ttr, y_tr).predict_proba(Tte)[:, 1]
        else:
            load = lambda ids: torch.from_numpy(np.stack(
                [np.load(os.path.join(RUN_DIR, CROPS[m], f"{s}.npy")) for s in ids])).to(device)
            Xtr, Xte = load(train), load(tst)
            per_seed = []
            for k in range(N_SEEDS):
                t1 = time.time()
                per_seed.append(predict_cnn(train_cnn(Xtr, y_tr, task, SEED + 1000 + k, device), Xte, device))
                probs[f"{m}_seed{k + 1}"] = per_seed[-1]
                print(f"  {m} seed {k + 1}/{N_SEEDS} ({time.time() - t1:.0f} s)", flush=True)
            probs[m] = np.mean(per_seed, axis=0)
            del Xtr, Xte
    probs["ensemble"] = np.mean([probs[m] for m in MEMBERS[task]], axis=0)

    pathology = test["pathology"].fillna("").to_numpy()
    patient = (test["group"] == "patient").to_numpy()
    hs = pathology == "HS"
    clusters = test["pair"].to_numpy() if task == "tle" else np.arange(len(tst))
    rows = []
    for name in ["ensemble"] + MEMBERS[task]:
        p = probs[name]
        right = (p >= 0.5) == y_te
        (a_lo, a_hi), (u_lo, u_hi) = bootstrap(y_te, p, clusters, np.random.default_rng(SEED))
        rows.append({"model": name, "primary": name == "ensemble", "n": len(y_te), **metrics(y_te, p),
                     "acc_ci_low": a_lo, "acc_ci_high": a_hi, "auc_ci_low": u_lo, "auc_ci_high": u_hi,
                     "acc_p_vs_chance": binomtest(int(right.sum()), len(right), 0.5, alternative="greater").pvalue,
                     "acc_HS_patients": right[hs].mean() if hs.any() else np.nan,
                     "acc_nonHS_patients": right[~hs & patient].mean() if (~hs & patient).any() else np.nan,
                     "n_HS": int(hs.sum()), "n_nonHS": int((~hs & patient).sum())})
    res = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    pred = test[["participant_id", "group", "pair", "tle_side", "pathology", "scanner"]].copy()
    pred["label"] = y_te
    for k, v in probs.items():
        pred[k] = v
    pred.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    res.to_csv(os.path.join(out_dir, "final_metrics.csv"), index=False)

    print(f"\nHELD-OUT TEST - {task} ({len(y_te)} subjects never used for development)")
    for _, r in res.iterrows():
        print(f"  {r['model']:<13}{' (PRIMARY)' if r['primary'] else '':<10} "
              f"acc {r['accuracy']:.3f} [{r['acc_ci_low']:.2f}-{r['acc_ci_high']:.2f}]  "
              f"AUC {r['auc']:.3f} [{r['auc_ci_low']:.2f}-{r['auc_ci_high']:.2f}]  "
              f"sens {r['sensitivity']:.3f} spec {r['specificity']:.3f} F1 {r['f1']:.3f} MCC {r['mcc']:.3f}  "
              f"p={r['acc_p_vs_chance']:.4f}  HS {r['acc_HS_patients']:.2f} (n={r['n_HS']}) "
              f"nonHS {r['acc_nonHS_patients']:.2f} (n={r['n_nonHS']})")
    print(f"saved -> {out_dir} | total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "tle"
    if task not in ("tle", "side"):
        sys.exit("task must be 'tle' or 'side'")
    main(task)
