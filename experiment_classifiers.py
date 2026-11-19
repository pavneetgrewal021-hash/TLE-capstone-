# ============================================
# EXPERIMENT: CLASSIFIERS x FEATURE COMBINATIONS (mentor's request)
# ============================================
# Compares 9 classifiers on many feature sets built from the pipeline's own
# outputs, for both questions, on the DEVELOPMENT subjects only (held-out
# test subjects are never loaded). Fixed default hyperparameters (nothing
# tuned on these results), scaler fitted inside each training fold.
#
# Feature sets (per subject):
#   gat_* / pca_*   from the Step 10 hemisphere vectors (16 numbers per side)
#     LR      left and right side by side (32)
#     avg     (left + right) / 2 (16)
#     diff    left - right (16)
#     absdiff |left - right| (16)
#     avg_absdiff  avg and |left - right| (32)
#     asymz   (left - right) as a z-score against CONTROLS (16); controls of
#             the training fold only (tle) / all development controls (side)
#   fusion_adaptive   Step 11 output (32)
#   tok_all / tok_T1 / tok_FLAIR   "no graph" baseline: per hemisphere mean
#             of the Step 7 token features (signed log), plus left - right
#
# CV: tle  = matched pairs kept together, 5 folds x 10 repeats
#     side = stratified 5 folds x 10 repeats
# Output: RUN_DIR/experiments/classifiers_<task>.csv (one row per
# feature set x classifier: mean and std of every metric), sorted by AUC.
#
# Run:  python experiment_classifiers.py tle    (or: side)

import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline_config import LABELS_FILE, RUN_DIR, dev_only

N_FOLDS, N_REPEATS, SEED = 5, int(os.environ.get("TLE_EXP_REPEATS", "10")), 42
N_JOBS = int(os.environ.get("TLE_EXP_JOBS", "8"))

CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=5000),
    "linear_svm": SVC(kernel="linear", probability=True),
    "rbf_svm": SVC(kernel="rbf", probability=True),
    "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=1),
    "gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200),
    "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "naive_bayes": GaussianNB(),
    "mlp": MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=2000),
}


class AsymZ(BaseEstimator, TransformerMixin):
    """X = [left | right]. Output (left - right - mean_c) / std_c where c are
    the control subjects: those with y == 0 in the training fold (tle), or a
    fixed reference passed in (side, whose samples are all patients)."""
    def __init__(self, ref=None):
        self.ref = ref

    def fit(self, X, y=None):
        h = X.shape[1] // 2
        d = self.ref if self.ref is not None else (X[:, :h] - X[:, h:])[np.asarray(y) == 0]
        self.mu_, self.sd_ = d.mean(0), d.std(0) + 1e-8
        return self

    def transform(self, X):
        h = X.shape[1] // 2
        return (X[:, :h] - X[:, h:] - self.mu_) / self.sd_


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load_features(subjects, control_ids):
    emb = os.path.join(RUN_DIR, "embedding_outputs")
    idx = pd.read_csv(os.path.join(emb, "graph_embedding_index.csv"))
    feats, needs_ref = {}, {}
    for src in ("gat", "pca"):
        E = np.load(os.path.join(emb, f"graph_embeddings_{src}.npy"))
        rows = {(s, h): i for i, (s, h) in enumerate(zip(idx["Subject_ID"], idx["Hemisphere"]))}
        get = lambda ids, h: np.stack([E[rows[(s, h)]] for s in ids])
        L, R = get(subjects, "L"), get(subjects, "R")
        feats[f"{src}_LR"] = np.hstack([L, R])
        feats[f"{src}_avg"] = (L + R) / 2
        feats[f"{src}_diff"] = L - R
        feats[f"{src}_absdiff"] = np.abs(L - R)
        feats[f"{src}_avg_absdiff"] = np.hstack([(L + R) / 2, np.abs(L - R)])
        feats[f"{src}_asymz"] = np.hstack([L, R])                       # transformed inside the fold
        needs_ref[f"{src}_asymz"] = get(control_ids, "L") - get(control_ids, "R")
    alpha = pd.read_csv(os.path.join(RUN_DIR, "fusion_outputs", "alpha_weights.csv"))
    fa = np.load(os.path.join(RUN_DIR, "fusion_outputs", "fusion_embeddings_adaptive.npy"))
    pos = {s: i for i, s in enumerate(alpha["Subject_ID"])}
    feats["fusion_adaptive"] = fa[[pos[s] for s in subjects]]

    tok = pd.read_csv(os.path.join(RUN_DIR, "token_outputs", "tokens.csv"))
    tok = tok[tok["Subject_ID"].isin(subjects)]
    cols = [c for c in tok.columns if c.startswith(("T1_", "FLAIR_"))]
    tok[cols] = signed_log(tok[cols].to_numpy())
    m = tok.groupby(["Subject_ID", "Hemisphere"])[cols].mean()
    for name, prefix in (("tok_all", ("T1_", "FLAIR_")), ("tok_T1", ("T1_",)), ("tok_FLAIR", ("FLAIR_",))):
        c = [x for x in cols if x.startswith(prefix)]
        L = m.xs("L", level="Hemisphere").loc[subjects, c].to_numpy()
        R = m.xs("R", level="Hemisphere").loc[subjects, c].to_numpy()
        feats[name] = np.hstack([(L + R) / 2, L - R])
    return feats, needs_ref


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else np.nan
    sens, spec, ppv, npv = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp), div(tn, tn + fn)
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens,
            "specificity": spec, "ppv": ppv, "npv": npv, "f1": div(2 * ppv * sens, ppv + sens),
            "auc": roc_auc_score(y, prob), "mcc": matthews_corrcoef(y, pred)}


def splits(y, groups, seed):
    if groups is not None:
        pairs = np.random.default_rng(seed).permutation(np.unique(groups))
        for test_pairs in np.array_split(pairs, N_FOLDS):
            m = np.isin(groups, test_pairs)
            yield np.where(~m)[0], np.where(m)[0]
    else:
        yield from StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def run_one(feat_name, X, y, groups, clf_name, task, ref):
    rows = []
    for r in range(N_REPEATS):
        prob = np.zeros(len(y))
        for tr, te in splits(y, groups, SEED + r):
            steps = [AsymZ(ref=None if task == "tle" else ref)] if ref is not None else []
            clf = clone(CLASSIFIERS[clf_name])
            if "random_state" in clf.get_params():
                clf.set_params(random_state=SEED + r)
            model = make_pipeline(*steps, StandardScaler(), clf).fit(X[tr], y[tr])
            prob[te] = model.predict_proba(X[te])[:, 1]
        rows.append(metrics(y, prob))
    df = pd.DataFrame(rows)
    out = {"features": feat_name, "classifier": clf_name, "n_features": X.shape[1] // (2 if ref is not None else 1)}
    for c in df.columns:
        out[c], out[c + "_std"] = df[c].mean(), df[c].std()
    return out


def main(task):
    t0 = time.time()
    lab = dev_only(pd.read_csv(LABELS_FILE, sep="\t"))
    controls = sorted(lab[(lab["group"] == "control")]["participant_id"])
    if task == "tle":
        lab = lab[lab["cohort"] == "matched"].sort_values("participant_id")
        y = (lab["group"] == "patient").astype(int).to_numpy()
        groups = lab["pair"].to_numpy()
    else:
        lab = lab[lab["group"] == "patient"].sort_values("participant_id")
        y = (lab["tle_side"] == "L").astype(int).to_numpy()
        groups = None
    subjects = lab["participant_id"].tolist()
    feats, refs = load_features(subjects, controls)
    vol_dir = os.environ.get("TLE_VOL_DIR")
    if vol_dir:                                                        # FreeSurfer volumes (experiment_volumetry.py)
        from experiment_volumetry import load_volumes
        vol = load_volumes(vol_dir).loc[subjects]
        if task == "tle":
            v = np.column_stack([vol[["hipp_L", "hipp_R", "amyg_L", "amyg_R"]].to_numpy(),
                                 vol[["hipp_AI", "amyg_AI"]].abs().to_numpy()])
            partners = ["pca_avg_absdiff", "tok_FLAIR", "gat_avg", "fusion_adaptive"]
        else:
            v = vol[["hipp_AI", "amyg_AI"]].to_numpy()
            partners = ["tok_T1", "gat_diff", "pca_diff", "tok_all"]
        feats["vol"] = v
        for p in partners:
            feats[f"vol+{p}"] = np.hstack([v, feats[p]])
    print(f"task={task} subjects={len(subjects)} feature_sets={len(feats)} classifiers={len(CLASSIFIERS)} "
          f"repeats={N_REPEATS} (loaded in {time.time() - t0:.0f} s)", flush=True)

    jobs = [(f, X, c) for f, X in feats.items() for c in CLASSIFIERS]
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_one)(f, X, y, groups, c, task, refs.get(f)) for f, X, c in jobs)
    out_dir = os.path.join(RUN_DIR, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results).sort_values("auc", ascending=False)
    path = os.path.join(out_dir, f"classifiers_{task}.csv")
    df.to_csv(path, index=False)
    show = ["features", "classifier", "accuracy", "accuracy_std", "auc", "auc_std", "sensitivity",
            "specificity", "f1", "mcc"]
    print(f"\nTOP 15 by AUC ({task}, mean over {N_REPEATS} repeats of {N_FOLDS}-fold CV):")
    print(df[show].head(15).round(3).to_string(index=False))
    print(f"\nBest per feature set:")
    print(df.groupby("features").head(1)[["features", "classifier", "accuracy", "auc"]].round(3).to_string(index=False))
    print(f"\nsaved -> {path} | {time.time() - t0:.0f} s")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "tle"
    if task not in ("tle", "side"):
        sys.exit("task must be 'tle' or 'side'")
    main(task)
