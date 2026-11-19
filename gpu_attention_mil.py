# ============================================
# GPU EXPERIMENT: ATTENTION-MIL ON BRAIN PATCHES
# ============================================
# Learns directly from the Step 5 patches instead of hand-made wavelet
# statistics. Each brain is a "bag" of ~350 patches (T1 + FLAIR, 32 mm):
#   1. a small 3D CNN turns every patch into a 128-number feature vector,
#      with the patch's position (x, y, z, hemisphere) added;
#   2. a gated attention layer (Ilse et al. 2018) gives every patch a weight
#      and averages the patch features with those weights;
#   3. a linear layer makes the decision for the whole brain.
# The attention weights show WHICH brain regions the model relied on.
#
# Tasks:  tle  = TLE patient vs healthy control (matched development pairs)
#         side = left vs right TLE (development patients; mirrored brains
#                with swapped labels are added during training)
# Evaluation: repeated 5-fold CV on the DEVELOPMENT subjects only (folds keep
# matched pairs together for "tle"); the held-out test subjects are never
# loaded. Fixed training schedule (no early stopping), so the test folds are
# never used for any choice.
#
# Run:   python gpu_attention_mil.py tle      (or: side)
#        TLE_MIL_SMOKE=1 python gpu_attention_mil.py tle   -> tiny CPU check
# Output: RUN_DIR/gpu_mil/<task>/  metrics, out-of-fold predictions, attention

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pipeline_config import LABELS_FILE, RUN_DIR, dev_only

SMOKE = os.environ.get("TLE_MIL_SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
N_REPEATS = 1 if SMOKE else int(os.environ.get("TLE_MIL_REPEATS", "3"))
EPOCHS = 2 if SMOKE else int(os.environ.get("TLE_MIL_EPOCHS", "40"))
BAGS_PER_BATCH = 4 if SMOKE else 8
TRAIN_PATCHES = 32 if SMOKE else 256      # random subset of each bag per training step
LR, WEIGHT_DECAY = 3e-4, 1e-2
SEED = 42
MID_X = 98                                 # MNI template x index of the midline (Step 5)
PATCH_DIR = os.path.join(RUN_DIR, "patch_outputs")


# ---------------------------------------------------------------- data
def load_bags(subjects):
    bags = {}
    for s in subjects:
        p = os.path.join(PATCH_DIR, s, "patches.npy")
        if not os.path.exists(p):
            print(f"  {s}: no patches, skipped")
            continue
        x = torch.from_numpy(np.load(p)).to(torch.float16)            # (N, 2, 32, 32, 32)
        meta = pd.read_csv(os.path.join(PATCH_DIR, s, "metadata.csv"))
        c = np.array([tuple(map(float, v.strip("()").split(","))) for v in meta["Center_Coordinate"]])
        pos = np.stack([(c[:, 0] - MID_X) / MID_X, c[:, 1] / 233 * 2 - 1, c[:, 2] / 189 * 2 - 1], axis=1)
        bags[s] = (x, torch.tensor(pos, dtype=torch.float32), meta["Patch_ID"].to_numpy())
    return bags


def mirror(x, pos):
    """Left-right mirror of a bag: flip every patch along x and negate x."""
    pos = pos.clone()
    pos[:, 0] = -pos[:, 0]
    return torch.flip(x, dims=[2]), pos


# ---------------------------------------------------------------- model
class PatchEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        def block(cin, cout, pool=True):
            layers = [nn.Conv3d(cin, cout, 3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True)]
            return layers + ([nn.MaxPool3d(2)] if pool else [])
        self.cnn = nn.Sequential(*block(2, 16), *block(16, 32), *block(32, 64), *block(64, dim, pool=False),
                                 nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.pos = nn.Sequential(nn.Linear(3, 16), nn.ReLU(inplace=True))
        self.mix = nn.Sequential(nn.Linear(dim + 16, dim), nn.ReLU(inplace=True), nn.Dropout(0.2))

    def forward(self, x, pos):
        return self.mix(torch.cat([self.cnn(x), self.pos(pos)], dim=1))


class AttentionMIL(nn.Module):
    def __init__(self, dim=128, att=64):
        super().__init__()
        self.enc = PatchEncoder(dim)
        self.V, self.U, self.w = nn.Linear(dim, att), nn.Linear(dim, att), nn.Linear(att, 1)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, pos, bag_sizes):
        h = self.enc(x, pos)                                           # (sum N, dim)
        scores = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(1)
        logits, weights = [], []
        for hb, sb in zip(torch.split(h, bag_sizes), torch.split(scores, bag_sizes)):
            a = torch.softmax(sb.float(), dim=0)
            logits.append(self.head((a.unsqueeze(1) * hb.float()).sum(0)))
            weights.append(a)
        return torch.cat(logits), weights


# ---------------------------------------------------------------- train / predict
def make_batch(bags, ids, labels, task, rng, train, device):
    """Bags already live on `device` (see main), so selection, mirroring and
    noise all run on the GPU - on the 15-CPU server, doing this on the CPU
    left the GPU idle."""
    xs, ps, ys, sizes = [], [], [], []
    for s in ids:
        x, pos, _ = bags[s]
        y = labels[s]
        if train:
            k = min(TRAIN_PATCHES, len(x))
            sel = torch.from_numpy(rng.choice(len(x), size=k, replace=False)).to(x.device)
            x, pos = x[sel], pos[sel]
            if rng.random() < 0.5:                                     # mirror augmentation
                x, pos = mirror(x, pos)
                if task == "side":
                    y = 1 - y                                          # a mirrored left-TLE brain looks right-sided
        xs.append(x); ps.append(pos); ys.append(y); sizes.append(len(x))
    x = torch.cat(xs).float()
    if train:
        x = x + 0.05 * torch.randn_like(x)                             # small intensity noise
    return x, torch.cat(ps), torch.tensor(ys, dtype=torch.float32, device=device), sizes


def train_model(bags, train_ids, labels, task, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AttentionMIL().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * int(np.ceil(len(train_ids) / BAGS_PER_BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    ytr = np.array([labels[s] for s in train_ids])
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device)
    model.train()
    for _ in range(EPOCHS):
        order = rng.permutation(train_ids)
        for i in range(0, len(order), BAGS_PER_BATCH):
            x, pos, y, sizes = make_batch(bags, order[i:i + BAGS_PER_BATCH], labels, task, rng, True, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = model(x.float(), pos, sizes)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict(model, bags, ids, labels, task, device):
    model.eval()
    probs, atts = [], {}
    for s in ids:
        x, pos, _, sizes = make_batch(bags, [s], labels, task, None, False, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logit, w = model(x.float(), pos, sizes)
        probs.append(torch.sigmoid(logit.float()).item())
        atts[s] = w[0].cpu().numpy()
    return np.array(probs), atts


# ---------------------------------------------------------------- metrics
def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else float("nan")
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    ppv, npv = div(tp, tp + fp), div(tn, tn + fn)
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": (sens + spec) / 2,
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "f1": div(2 * ppv * sens, ppv + sens), "auc": roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan"),
            "mcc": matthews_corrcoef(y, pred)}


def folds(ids, labels, groups, seed):
    y = np.array([labels[s] for s in ids])
    if groups is not None:                                             # whole matched pairs per fold
        pairs = np.random.default_rng(seed).permutation(np.unique(groups))
        for test_pairs in np.array_split(pairs, N_FOLDS):
            m = np.isin(groups, test_pairs)
            yield np.where(~m)[0], np.where(m)[0]
    else:
        yield from StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(ids, y)


# ---------------------------------------------------------------- main
def main(task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(14, os.cpu_count() or 1))
    out_dir = os.path.join(RUN_DIR, "gpu_mil", task + ("_smoke" if SMOKE else ""))
    os.makedirs(out_dir, exist_ok=True)

    lab = dev_only(pd.read_csv(LABELS_FILE, sep="\t"))
    if task == "tle":
        lab = lab[lab["cohort"] == "matched"]
        labels = dict(zip(lab["participant_id"], (lab["group"] == "patient").astype(int)))
    else:
        lab = lab[lab["group"] == "patient"]
        labels = dict(zip(lab["participant_id"], (lab["tle_side"] == "L").astype(int)))
    print(f"task={task} device={device} subjects={len(labels)} folds={N_FOLDS} repeats={N_REPEATS} epochs={EPOCHS}")

    t0 = time.time()
    wanted = sorted(labels)
    if SMOKE:                                                          # 8 complete pairs / 16 patients
        wanted = (sorted(lab[lab["pair"].isin(sorted(lab["pair"].unique())[:8])]["participant_id"])
                  if task == "tle" else wanted[:16])
    bags = load_bags(wanted)
    # all patches together are a few GB in float16: keep them on the GPU
    bags = {s: (x.to(device), pos.to(device), pid) for s, (x, pos, pid) in bags.items()}
    ids = np.array(sorted(bags))
    groups = np.array([lab.set_index("participant_id").loc[s, "pair"] for s in ids]) if task == "tle" else None
    print(f"loaded {len(ids)} bags, {sum(len(bags[s][0]) for s in ids)} patches in {time.time() - t0:.0f} s")

    y = np.array([labels[s] for s in ids])
    rows, oof_rows, att_sum, att_n = [], [], {}, {}
    for r in range(N_REPEATS):
        prob = np.zeros(len(ids))
        for k, (tr, te) in enumerate(folds(ids, labels, groups, SEED + r)):
            t1 = time.time()
            model = train_model(bags, ids[tr], labels, task, SEED + 100 * r + k, device)
            p, atts = predict(model, bags, ids[te], labels, task, device)
            prob[te] = p
            for s, a in atts.items():
                att_sum[s] = att_sum.get(s, 0) + a
                att_n[s] = att_n.get(s, 0) + 1
            print(f"  repeat {r + 1} fold {k + 1}: trained on {len(tr)}, tested {len(te)} ({time.time() - t1:.0f} s)", flush=True)
        m = metrics(y, prob)
        rows.append({"repeat": r + 1, **m})
        oof_rows += [{"repeat": r + 1, "subject": s, "label": int(yy), "prob": float(pp)} for s, yy, pp in zip(ids, y, prob)]
        print(f"repeat {r + 1}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(out_dir, "per_repeat_metrics.csv"), index=False)
    summary = per.drop(columns="repeat").agg(["mean", "std"]).T
    summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"))
    pd.DataFrame(oof_rows).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    att_rows = []
    for s in att_sum:
        _, pos, pid = bags[s]
        pos = pos.cpu().numpy()
        for i, a in enumerate(att_sum[s] / att_n[s]):
            att_rows.append({"subject": s, "patch_id": int(pid[i]), "x": float(pos[i, 0]), "y": float(pos[i, 1]),
                             "z": float(pos[i, 2]), "attention": float(a)})
    pd.DataFrame(att_rows).to_csv(os.path.join(out_dir, "attention_weights.csv"), index=False)
    print("\nSUMMARY (mean +- std over repeats)")
    print(summary.round(3).to_string())
    print(f"saved -> {out_dir} | total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "tle"
    if task not in ("tle", "side"):
        sys.exit("task must be 'tle' or 'side'")
    main(task)
