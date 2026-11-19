from __future__ import annotations

# ============================================
# STEP 13: LATERALIZATION (LEFT vs RIGHT TLE)
# ============================================
# Step 12 answers "TLE or healthy?". This step adds the second question a
# surgeon needs: "which side?". Together they form a two-stage system:
#
#   MRI -> Stage 1 (Step 12): patient or control?
#            -> control              => "healthy"
#            -> patient -> Stage 2 (this step): left or right TLE?
#
# Trained on PATIENTS ONLY (label = Op_Side from the IDEAS metadata, stored
# as tle_side in labels.tsv). Uses the hemisphere vectors from Step 10, so
# nothing in Steps 0-12 changes.
#
# FEATURES - control-normalised asymmetry:
#   d = h_L - h_R (left minus right hemisphere vector, 16 numbers).
#   Healthy brains are not perfectly symmetric, so d of a healthy person is
#   not zero. We estimate that NORMAL asymmetry from the controls (mean mu,
#   spread sigma per dimension) and use z = (d - mu) / sigma: "how unusual is
#   this person's left-right difference compared with healthy people". In a
#   left-TLE patient the abnormality sits on the left, so z should point the
#   opposite way to a right-TLE patient. Controls carry no side label, so
#   using them for mu/sigma leaks nothing about the side.
#
# MIRROR AUGMENTATION:
#   Flipping a brain left-to-right turns a left-TLE brain into (roughly) a
#   right-TLE brain. In z-space the flip is simply z -> -z (the abnormal part
#   changes side; normal anatomy is already removed by subtracting mu). So
#   every training patient is also added mirrored with the label flipped,
#   doubling the training data - only in TRAINING folds, never in testing.
#
# PRIMARY CONFIGURATION: see PRIMARY below. First run (50 patients) used
# GAT vectors (not significant); the current primary - PCA vectors +
# control-normalised asymmetry + mirror augmentation + L2 logistic
# regression - was pre-registered before the extra patients were processed
# and is checked with a train-on-discovery / test-on-new confirmation test.
# Other combinations are reported only as an ablation.
#
# EVALUATION (patients only, n=50): leave-one-out, repeated stratified
# 5-fold x10, permutation test (1000 shuffles), bootstrap 95% CI; accuracy
# split by pathology (HS vs non-HS). Plus a combined three-way result
# (healthy / left TLE / right TLE) using Step 12's leave-one-out Stage-1
# predictions.

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.svm import SVC

from pipeline_config import LABELS_FILE, RUN_DIR, dev_only

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(RUN_DIR)
EMBEDDING_DIR = BASE_DIR / "embedding_outputs"
STAGE1_PREDICTIONS = BASE_DIR / "classifier_outputs" / "predictions.csv"
OUTPUT_DIR = BASE_DIR / "lateralization_outputs"
FIGURES_DIR = BASE_DIR / "figures" / "step13"
METRICS_DIR = BASE_DIR / "metrics" / "step13"
LOGS_DIR = BASE_DIR / "logs"

STAGE1_MODEL = "linear_svm"   # best Step 12 model (nested LOOCV 77%, AUC 0.83)
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
CV_N_SPLITS = 5
CV_N_REPEATS = 10
RANDOM_SEED = 42
DPI = 300

MODELS = {
    "logistic_regression": LogisticRegression(C=1.0, max_iter=2000),
    "linear_svm": SVC(kernel="linear", C=1.0, probability=True),
}
# PRE-REGISTERED PRIMARY CONFIGURATION (fixed 2026-10-08, BEFORE the extra
# patients were processed). First run (50 patients): the GAT-based primary
# was not significant (AUC 0.59, p=0.18), but all 12 PCA-based variants
# reached AUC ~0.65-0.72. Because that was spotted after looking at 24
# variants, it is only a hypothesis - so it is fixed here in advance and
# tested on patients the idea was NOT derived from (confirmation test below).
# Rationale: the GAT is trained (Step 10) to give the same vector for
# perturbed copies of a graph, which plausibly teaches it to ignore small
# left-right differences; the PCA summary keeps them.
PRIMARY = {"source": "pca", "features": "asym_normalised", "mirror": True, "model": "logistic_regression"}

# The 50 patients the PCA hypothesis was formed on. Confirmation test: train
# on exactly these, test on every other patient (never seen when the
# hypothesis was formed).
DISCOVERY_PATIENTS_FILE = Path(__file__).resolve().parent / "lateralization_discovery_patients.txt"
N_CONFIRMATION_PERMUTATIONS = 10000


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step13_lateralization")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_path, mode="w"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_hemisphere_vectors(source: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Left and right hemisphere vectors per subject from Step 10
    ("gat" = graph attention network, "pca" = PCA baseline)."""
    index = pd.read_csv(EMBEDDING_DIR / "graph_embedding_index.csv")
    emb = np.load(EMBEDDING_DIR / f"graph_embeddings_{source}.npy")
    rows = {(s, h): i for i, (s, h) in enumerate(zip(index["Subject_ID"], index["Hemisphere"]))}
    subjects = sorted({s for s, _ in rows if (s, "L") in rows and (s, "R") in rows})
    h_l = np.stack([emb[rows[(s, "L")]] for s in subjects])
    h_r = np.stack([emb[rows[(s, "R")]] for s in subjects])
    return subjects, h_l, h_r


def normative_asymmetry(d: np.ndarray, d_controls: np.ndarray) -> np.ndarray:
    """z-score each asymmetry dimension against healthy controls."""
    mu = d_controls.mean(axis=0)
    sigma = d_controls.std(axis=0) + 1e-8
    return (d - mu) / sigma


def build_features(kind: str, h_l: np.ndarray, h_r: np.ndarray, d_controls: np.ndarray) -> np.ndarray:
    if kind == "asym_normalised":
        return normative_asymmetry(h_l - h_r, d_controls)
    if kind == "asym_raw":
        return h_l - h_r
    if kind == "concat":
        return np.concatenate([h_l, h_r], axis=1)
    raise ValueError(kind)


def mirror(kind: str, X: np.ndarray) -> np.ndarray:
    """Left-right flipped version of the features (labels get flipped too)."""
    if kind in ("asym_normalised", "asym_raw"):
        return -X
    half = X.shape[1] // 2
    return np.concatenate([X[:, half:], X[:, :half]], axis=1)


# ---------------------------------------------------------------------------
# Training / evaluation (y: 1 = LEFT TLE, 0 = RIGHT TLE)
# ---------------------------------------------------------------------------
def fit(model, X: np.ndarray, y: np.ndarray, kind: str, use_mirror: bool, seed: int):
    if use_mirror:
        X = np.concatenate([X, mirror(kind, X)])
        y = np.concatenate([y, 1 - y])
    m = clone(model)
    if hasattr(m, "random_state"):
        m.set_params(random_state=seed)
    return m.fit(X, y)


def out_of_fold(X, y, splits, model, kind, use_mirror, seed=RANDOM_SEED) -> np.ndarray:
    """Probability of LEFT for every patient, each predicted by a model that
    never saw it."""
    prob = np.zeros(len(y))
    for train_idx, test_idx in splits:
        m = fit(model, X[train_idx], y[train_idx], kind, use_mirror, seed)
        prob[test_idx] = m.predict_proba(X[test_idx])[:, 1]
    return prob


def loocv_prob(X, y, model, kind, use_mirror) -> np.ndarray:
    return out_of_fold(X, y, LeaveOneOut().split(X), model, kind, use_mirror)


def repeated_cv(X, y, model, kind, use_mirror) -> pd.DataFrame:
    rows = []
    for r in range(CV_N_REPEATS):
        cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_SEED + r)
        prob = out_of_fold(X, y, cv.split(X, y), model, kind, use_mirror, RANDOM_SEED + r)
        rows.append({"repeat": r, "accuracy": accuracy_score(y, prob >= 0.5), "auc": roc_auc_score(y, prob)})
    return pd.DataFrame(rows)


def permutation_p(X, y, model, kind, use_mirror, observed_auc: float, n: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    null = np.array([roc_auc_score(yp, loocv_prob(X, yp, model, kind, use_mirror))
                     for yp in (rng.permutation(y) for _ in range(n))])
    return float((np.sum(null >= observed_auc) + 1) / (n + 1)), null


def bootstrap_ci(y, prob) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    accs, aucs = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        accs.append(accuracy_score(y[idx], prob[idx] >= 0.5))
        aucs.append(roc_auc_score(y[idx], prob[idx]))
    return {"accuracy_ci_low": np.percentile(accs, 2.5), "accuracy_ci_high": np.percentile(accs, 97.5),
            "auc_ci_low": np.percentile(aucs, 2.5), "auc_ci_high": np.percentile(aucs, 97.5)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for d in (OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOGS_DIR / "step13.log")
    logger.info("Starting Step 13: Lateralization (left vs right TLE)")
    t0 = time.perf_counter()

    labels = dev_only(pd.read_csv(LABELS_FILE, sep="\t")).set_index("participant_id")  # held-out test subjects excluded

    ablation_rows = []
    primary = None
    for source in ("gat", "pca"):
        subjects, h_l, h_r = load_hemisphere_vectors(source)
        # Embeddings can include leftover subjects from an earlier cohort
        # (their patch folders still exist); keep only the current cohort.
        keep = np.isin(subjects, labels.index)
        if not keep.all():
            logger.warning("%s: ignoring %d subject(s) not in labels.tsv: %s", source,
                           (~keep).sum(), list(np.array(subjects)[~keep]))
        subjects, h_l, h_r = list(np.array(subjects)[keep]), h_l[keep], h_r[keep]
        meta = labels.loc[subjects]
        is_patient = (meta["group"] == "patient").to_numpy()
        is_control = ~is_patient
        d_controls = (h_l - h_r)[is_control]
        pat_ids = list(np.array(subjects)[is_patient])
        y = (meta.loc[pat_ids, "tle_side"] == "L").astype(int).to_numpy()

        for kind in ("asym_normalised", "asym_raw", "concat"):
            X = build_features(kind, h_l[is_patient], h_r[is_patient], d_controls)
            for use_mirror in (True, False):
                for model_name, model in MODELS.items():
                    prob = loocv_prob(X, y, model, kind, use_mirror)
                    rep = repeated_cv(X, y, model, kind, use_mirror)
                    is_primary = (source, kind, use_mirror, model_name) == (
                        PRIMARY["source"], PRIMARY["features"], PRIMARY["mirror"], PRIMARY["model"])
                    ablation_rows.append({
                        "Source": source, "Features": kind, "Mirror_Augmentation": use_mirror,
                        "Model": model_name, "Primary": is_primary,
                        "LOOCV_Accuracy": accuracy_score(y, prob >= 0.5), "LOOCV_AUC": roc_auc_score(y, prob),
                        "RepCV_Accuracy_Mean": rep["accuracy"].mean(), "RepCV_Accuracy_Std": rep["accuracy"].std(),
                        "RepCV_AUC_Mean": rep["auc"].mean(), "RepCV_AUC_Std": rep["auc"].std(),
                    })
                    if is_primary:
                        primary = {"subjects": subjects, "h_l": h_l, "h_r": h_r, "is_patient": is_patient,
                                   "d_controls": d_controls, "pat_ids": pat_ids, "X": X, "y": y,
                                   "prob": prob, "rep": rep, "model": model, "kind": kind}

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(METRICS_DIR / "ablation.csv", index=False)

    # --- Primary configuration: full statistics ---
    p = primary
    X, y, prob = p["X"], p["y"], p["prob"]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, prob)
    logger.info("Primary LOOCV: accuracy=%.3f auc=%.3f (n=%d patients, %d left / %d right)",
                accuracy_score(y, pred), auc, len(y), int(y.sum()), int(len(y) - y.sum()))
    perm_p, null = permutation_p(X, y, p["model"], p["kind"], PRIMARY["mirror"], auc, N_PERMUTATIONS)
    logger.info("Permutation test (%d shuffles): null mean AUC=%.3f, p=%.4f", N_PERMUTATIONS, null.mean(), perm_p)
    ci = bootstrap_ci(y, prob)

    pathology = labels.loc[p["pat_ids"], "pathology"].to_numpy()
    patient_df = pd.DataFrame({
        "Subject_ID": p["pat_ids"], "True_Side": np.where(y == 1, "L", "R"),
        "Prob_Left": prob, "Predicted_Side": np.where(pred == 1, "L", "R"),
        "Correct": pred == y, "Pathology": pathology,
    })
    patient_df.to_csv(OUTPUT_DIR / "lateralization_predictions_patients.csv", index=False)
    by_path = patient_df.assign(HS=np.where(patient_df["Pathology"] == "HS", "HS", "non-HS")) \
        .groupby("HS")["Correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy", "count": "n"})
    by_path.to_csv(METRICS_DIR / "accuracy_by_pathology.csv")

    pd.DataFrame([{
        "n_patients": len(y), "n_left": int(y.sum()), "n_right": int(len(y) - y.sum()),
        "loocv_accuracy": accuracy_score(y, pred), "loocv_auc": auc,
        "sensitivity_left": tp / (tp + fn), "sensitivity_right": tn / (tn + fp),
        **ci, "permutation_p_auc": perm_p, "null_mean_auc": float(null.mean()),
        "repcv_accuracy_mean": p["rep"]["accuracy"].mean(), "repcv_accuracy_std": p["rep"]["accuracy"].std(),
        "repcv_auc_mean": p["rep"]["auc"].mean(), "repcv_auc_std": p["rep"]["auc"].std(),
    }]).to_csv(METRICS_DIR / "primary_result.csv", index=False)

    # --- Confirmation test of the pre-registered configuration ---
    # Train on the 50 discovery patients only; test on all other patients.
    discovery = set(DISCOVERY_PATIENTS_FILE.read_text().split()) if DISCOVERY_PATIENTS_FILE.exists() else set()
    is_disc = np.array([s in discovery for s in p["pat_ids"]])
    if is_disc.sum() >= 10 and (~is_disc).sum() >= 10:
        conf_model = fit(p["model"], X[is_disc], y[is_disc], p["kind"], PRIMARY["mirror"], RANDOM_SEED)
        conf_prob = conf_model.predict_proba(X[~is_disc])[:, 1]
        y_test = y[~is_disc]
        conf_auc = roc_auc_score(y_test, conf_prob)
        rng = np.random.default_rng(RANDOM_SEED)
        conf_null = np.array([roc_auc_score(rng.permutation(y_test), conf_prob)
                              for _ in range(N_CONFIRMATION_PERMUTATIONS)])
        conf_p = float((np.sum(conf_null >= conf_auc) + 1) / (N_CONFIRMATION_PERMUTATIONS + 1))
        conf_pred = (conf_prob >= 0.5).astype(int)
        test_path = labels.loc[np.array(p["pat_ids"])[~is_disc], "pathology"].to_numpy()
        pd.DataFrame([{
            "train_patients": int(is_disc.sum()), "test_patients": int((~is_disc).sum()),
            "test_left": int(y_test.sum()), "test_right": int(len(y_test) - y_test.sum()),
            "test_accuracy": accuracy_score(y_test, conf_pred), "test_auc": conf_auc,
            "permutation_p_auc": conf_p,
            "test_accuracy_HS": float((conf_pred == y_test)[test_path == "HS"].mean()),
            "test_accuracy_nonHS": float((conf_pred == y_test)[test_path != "HS"].mean()),
        }]).to_csv(METRICS_DIR / "confirmation_test.csv", index=False)
        logger.info("CONFIRMATION (train %d discovery patients -> test %d new): accuracy=%.3f auc=%.3f p=%.4f",
                    is_disc.sum(), (~is_disc).sum(), accuracy_score(y_test, conf_pred), conf_auc, conf_p)
    else:
        logger.warning("Confirmation test skipped: need >=10 discovery and >=10 new patients "
                       "(have %d / %d).", is_disc.sum(), (~is_disc).sum())

    # --- Combined three-way system: healthy / left TLE / right TLE ---
    # Stage 1 = Step 12's leave-one-out predictions. Stage 2 for a TRUE
    # patient = its own leave-one-out side prediction above; for a control
    # that Stage 1 wrongly flags, a side model trained on all 50 patients
    # (that control was never in the patient training set).
    stage1 = pd.read_csv(STAGE1_PREDICTIONS).set_index("Subject_ID")
    full_model = fit(p["model"], X, y, p["kind"], PRIMARY["mirror"], RANDOM_SEED)
    side_of_patient = dict(zip(p["pat_ids"], np.where(pred == 1, "L", "R")))
    control_ids = [s for s, is_pat in zip(p["subjects"], p["is_patient"]) if not is_pat]
    combined = []
    for i, s in enumerate(p["subjects"]):
        if s not in stage1.index:
            continue  # lateralization-only patients are not part of Stage 1
        true = "healthy" if labels.loc[s, "group"] == "control" else f"TLE-{labels.loc[s, 'tle_side']}"
        if stage1.loc[s, f"{STAGE1_MODEL}_pred"] == 0:
            predicted = "healthy"
        elif s in side_of_patient:
            predicted = f"TLE-{side_of_patient[s]}"
        else:
            # this control's own asymmetry must not be part of the "normal"
            # reference it is compared against
            others = np.array([c != s for c in control_ids])
            x = build_features(p["kind"], p["h_l"][i:i + 1], p["h_r"][i:i + 1], p["d_controls"][others])
            predicted = "TLE-L" if full_model.predict_proba(x)[0, 1] >= 0.5 else "TLE-R"
        combined.append({"Subject_ID": s, "True": true, "Predicted": predicted})
    combined_df = pd.DataFrame(combined)
    combined_df.to_csv(OUTPUT_DIR / "three_way_predictions.csv", index=False)
    classes = ["healthy", "TLE-L", "TLE-R"]
    cm3 = confusion_matrix(combined_df["True"], combined_df["Predicted"], labels=classes)
    pd.DataFrame(cm3, index=[f"true {c}" for c in classes], columns=[f"pred {c}" for c in classes]) \
        .to_csv(METRICS_DIR / "three_way_confusion.csv")
    three_way_acc = float(np.trace(cm3) / cm3.sum())
    logger.info("Three-way (healthy / TLE-L / TLE-R) accuracy: %.3f", three_way_acc)

    # --- Figures ---
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, label=f"left vs right (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="chance")
    ax.set_xlabel("False positive rate (right called left)")
    ax.set_ylabel("True positive rate (left called left)")
    ax.set_title("Lateralization ROC (LOOCV, 50 patients)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "lateralization_roc.png", dpi=DPI); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm3, cmap="Blues")
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(cm3[r, c]), ha="center", va="center", fontsize=12)
    ax.set_xticks(range(3)); ax.set_xticklabels(classes)
    ax.set_yticks(range(3)); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Two-stage system, three-way accuracy {three_way_acc:.0%}")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "three_way_confusion.png", dpi=DPI); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=40, color="grey", alpha=0.7, label="shuffled labels")
    ax.axvline(auc, color="red", label=f"real AUC {auc:.2f} (p={perm_p:.3f})")
    ax.set_xlabel("LOOCV AUC"); ax.set_ylabel("count"); ax.legend()
    ax.set_title("Permutation test")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "permutation_null.png", dpi=DPI); plt.close(fig)

    logger.info("Step 13 complete | Total runtime=%.1f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
