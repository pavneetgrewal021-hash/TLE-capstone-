# ============================================
# GPU EXPERIMENT: 3D CNN ON THE HIPPOCAMPUS + AMYGDALA BOXES
# ============================================
# Input: hippo_crops.py boxes (left, right-mirrored; T1 + FLAIR; 48x64x48 mm).
# One shared 3D CNN turns each side into a feature vector fL, fR.
#   tle  : head on [ (fL + fR) / 2 , |fL - fR| ]  -> patient vs control
#   side : logit = w . (fL - fR)  (no bias), so swapping the two sides flips
#          the prediction exactly - left and right are treated symmetrically
# Training augmentation: small random shifts, intensity scaling and noise,
# and swapping the two sides (side task: with the label flipped).
#
# Every fold also fits the current best tabular model on the SAME training
# subjects and reports three things on the test fold:
#   cnn       the CNN alone
#   tabular   tle: LDA on volumes + PCA graph features (avg, |L-R|)
#             side: logistic regression on hippocampus/amygdala asymmetry
#   ensemble  plain average of the two probabilities (no weights fitted)
# Development subjects only; same folds as experiment_classifiers.py
# (tle: matched pairs together; side: stratified), 5 folds x N repeats.
#
# Run:  TLE_VOL_DIR=fs_tables python gpu_hippo_cnn.py tle   (or side)
#       TLE_MIL_SMOKE=1 ...  -> tiny CPU check

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_classifiers import load_features, metrics, splits
from experiment_volumetry import load_volumes
from pipeline_config import LABELS_FILE, RUN_DIR, dev_only

SMOKE = os.environ.get("TLE_MIL_SMOKE") == "1"
N_REPEATS = 1 if SMOKE else int(os.environ.get("TLE_CNN_REPEATS", "5"))
EPOCHS = 2 if SMOKE else int(os.environ.get("TLE_CNN_EPOCHS", "80"))
BATCH, WEIGHT_DECAY, SEED = 16, 5e-2, 42
LR_TAG = os.environ.get("TLE_CNN_LR")                                       # learning-rate sweep (lr_sweep_summary.py)
LR = float(LR_TAG) if LR_TAG else 1e-3
CROP_NAME = os.environ.get("TLE_CROP_DIR", "hippo_crops")                 # or temporal_crops
CROP_DIR = os.path.join(RUN_DIR, CROP_NAME)


class Encoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        def block(cin, cout):
            return [nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True),
                    nn.LeakyReLU(0.1, inplace=True), nn.MaxPool3d(2)]
        self.net = nn.Sequential(*block(2, 16), *block(16, 32), *block(32, 64), *block(64, dim),
                                 nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3))

    def forward(self, x):
        return self.net(x)


class HippoNet(nn.Module):
    def __init__(self, task, dim=64):
        super().__init__()
        self.task, self.enc = task, Encoder(dim)
        self.head = nn.Sequential(nn.Linear(2 * dim, 32), nn.LeakyReLU(0.1), nn.Dropout(0.3), nn.Linear(32, 1)) \
            if task == "tle" else nn.Linear(dim, 1, bias=False)

    def forward(self, x):                                              # x: (B, 2 sides, 2 ch, D, H, W)
        b = x.shape[0]
        f = self.enc(x.flatten(0, 1)).view(b, 2, -1)
        fl, fr = f[:, 0], f[:, 1]
        if self.task == "tle":
            return self.head(torch.cat([(fl + fr) / 2, (fl - fr).abs()], 1)).squeeze(1)
        return self.head(fl - fr).squeeze(1)


def augment(x, y, task, gen):
    b = x.shape[0]
    shift = torch.randint(-3, 4, (3,), generator=gen).tolist()          # same small shift for the batch
    x = torch.roll(x, shifts=shift, dims=(3, 4, 5))
    x = x * (1 + 0.1 * torch.randn(b, 1, 2, 1, 1, 1, generator=gen, device="cpu").to(x.device))
    x = x + 0.05 * torch.randn_like(x)
    swap = torch.rand(b, generator=gen) < 0.5
    if swap.any():
        idx = swap.nonzero().squeeze(1).to(x.device)
        x[idx] = x[idx].flip(1)                                         # exchange left and right boxes
        if task == "side":
            y = y.clone()
            y[idx] = 1 - y[idx]
    return x, y


def train_cnn(X, y, task, seed, device):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = HippoNet(task).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * int(np.ceil(len(y) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y), BATCH):
            b = order[i:i + BATCH].to(device)
            xb, yb = augment(X[b].float(), yt[b], task, gen)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict_cnn(model, X, device):
    model.eval()
    out = []
    for i in range(0, len(X), BATCH):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(X[i:i + BATCH].float()).float()).cpu())
    return torch.cat(out).numpy()


def main(task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    t0 = time.time()
    lab = dev_only(pd.read_csv(LABELS_FILE, sep="\t"))
    controls = sorted(lab[lab["group"] == "control"]["participant_id"])
    if task == "tle":
        lab = lab[lab["cohort"] == "matched"].sort_values("participant_id")
        if SMOKE:
            lab = lab[lab["pair"].isin(sorted(lab["pair"].unique())[:8])]
        y = (lab["group"] == "patient").astype(int).to_numpy()
        groups = lab["pair"].to_numpy()
    else:
        lab = lab[lab["group"] == "patient"].sort_values("participant_id")
        if SMOKE:
            lab = lab.head(16)
        y = (lab["tle_side"] == "L").astype(int).to_numpy()
        groups = None
    subjects = lab["participant_id"].tolist()
    pathology = lab["pathology"].fillna("").to_numpy()

    X = torch.from_numpy(np.stack([np.load(os.path.join(CROP_DIR, f"{s}.npy")) for s in subjects])).to(device)
    vol = load_volumes(os.environ.get("TLE_VOL_DIR", "fs_tables")).loc[subjects]
    if task == "tle":
        feats, _ = load_features(subjects, controls)
        T = np.hstack([vol[["hipp_L", "hipp_R", "amyg_L", "amyg_R"]].to_numpy(),
                       vol[["hipp_AI", "amyg_AI"]].abs().to_numpy(), feats["pca_avg_absdiff"]])
        tab_model = lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    else:
        T = vol[["hipp_AI", "amyg_AI"]].to_numpy()
        tab_model = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    print(f"task={task} device={device} subjects={len(subjects)} crops={tuple(X.shape)} repeats={N_REPEATS} "
          f"epochs={EPOCHS} lr={LR:g} (loaded in {time.time() - t0:.0f} s)", flush=True)

    rows, oof = [], []
    for r in range(N_REPEATS):
        p_cnn, p_tab = np.zeros(len(y)), np.zeros(len(y))
        for k, (tr, te) in enumerate(splits(y, groups, SEED + r)):
            t1 = time.time()
            model = train_cnn(X[tr], y[tr], task, SEED + 100 * r + k, device)
            p_cnn[te] = predict_cnn(model, X[te], device)
            p_tab[te] = tab_model().fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
            print(f"  repeat {r + 1} fold {k + 1}: {len(tr)} train / {len(te)} test ({time.time() - t1:.0f} s)", flush=True)
        p_ens = (p_cnn + p_tab) / 2
        for name, p in (("cnn", p_cnn), ("tabular", p_tab), ("ensemble", p_ens)):
            m = metrics(y, p)
            hs = pathology == "HS"
            rows.append({"repeat": r + 1, "model": name, **m,
                         "acc_HS_patients": np.mean(((p >= 0.5) == y)[hs]) if hs.any() else np.nan,
                         "acc_nonHS_patients": np.mean(((p >= 0.5) == y)[(~hs) & (lab["group"] == "patient").to_numpy()])})
        oof += [{"repeat": r + 1, "subject": s, "label": int(a), "cnn": b, "tabular": c, "ensemble": d}
                for s, a, b, c, d in zip(subjects, y, p_cnn, p_tab, p_ens)]
        print(f"repeat {r + 1}: " + " | ".join(
            f"{x['model']} acc={x['accuracy']:.3f} auc={x['auc']:.3f}" for x in rows[-3:]), flush=True)

    out_dir = os.path.join(RUN_DIR, "gpu_hippo", CROP_NAME,
                           task + (f"_lr{LR_TAG}" if LR_TAG else "") + ("_smoke" if SMOKE else ""))
    os.makedirs(out_dir, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(out_dir, "per_repeat_metrics.csv"), index=False)
    pd.DataFrame(oof).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    summary = per.drop(columns="repeat").groupby("model").agg(["mean", "std"])
    summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"))
    keep = ["accuracy", "auc", "sensitivity", "specificity", "f1", "mcc", "acc_HS_patients", "acc_nonHS_patients"]
    print("\nSUMMARY (mean over repeats)")
    print(per.groupby("model")[keep].mean().round(3).to_string())
    print(f"saved -> {out_dir} | total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "tle"
    if task not in ("tle", "side"):
        sys.exit("task must be 'tle' or 'side'")
    main(task)
