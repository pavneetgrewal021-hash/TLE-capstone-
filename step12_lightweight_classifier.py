from __future__ import annotations

import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from pipeline_config import LABELS_FILE, RUN_DIR, dev_only

BASE_DIR = Path(RUN_DIR)
FUSION_INPUT_DIR = BASE_DIR / "fusion_outputs"
CLASSIFIER_OUTPUT_DIR = BASE_DIR / "classifier_outputs"
FIGURES_DIR = BASE_DIR / "figures" / "final"
METRICS_DIR = BASE_DIR / "metrics" / "final"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# fusion_outputs/ has no diagnosis labels. They come from labels.tsv,
# written by step0_select_subjects.py from the IDEAS metadata (IDs
# >= 4001 are controls). NOT Capstone/code/participants.tsv - that file
# belongs to a different dataset (UNAM ds004469) and matched none of the
# IDEAS subjects, which is why the first run dropped all 75.
PARTICIPANTS_FILE_CANDIDATES = [Path(LABELS_FILE)]
PARTICIPANT_ID_COLUMN_CANDIDATES = ["participant_id", "Subject_ID", "subject_id"]
LABEL_COLUMN_CANDIDATES = ["group", "Group", "diagnosis", "Diagnosis", "patient_status", "label", "dx_group", "epilepsy_diagnosis"]
LABEL_COLUMN_OVERRIDE: Optional[str] = None
POSITIVE_CLASS_VALUES = ["patient", "epilepsy", "pat", "case", "tle", "1", "true", "yes"]

INNER_CV_FOLDS = 3
N_PERMUTATIONS = 100
N_SEEDS = 10
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# Repeated 80/20 evaluation alongside LOOCV: 5-fold CV repeated 10 times
# with different shuffles. Folds are grouped by matched PAIR (labels.tsv
# "pair"), so a patient and its scanner/sex/age-matched control are always
# on the same side of the split - otherwise the test fold's composition
# could differ from the training fold in scanner/age mix.
CV_N_SPLITS = 5
CV_N_REPEATS = 10
DPI = 300

MODEL_SPECS = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000),
        "param_grid": {"C": [0.01, 0.1, 1.0, 10.0]},
    },
    "linear_svm": {
        "estimator": SVC(kernel="linear", probability=True),
        "param_grid": {"C": [0.01, 0.1, 1.0, 10.0]},
    },
    "mlp": {
        "estimator": MLPClassifier(hidden_layer_sizes=(16,), max_iter=300),
        "param_grid": {"alpha": [1e-4, 1e-3, 1e-2, 1e-1]},
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step12_lightweight_classifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_fusion_embeddings(fusion_dir: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    alpha_path = fusion_dir / "alpha_weights.csv"
    if not alpha_path.exists():
        raise FileNotFoundError(f"Missing {alpha_path} (expected from Step 11).")
    index_df = pd.read_csv(alpha_path)
    subject_ids = index_df["Subject_ID"].astype(str).tolist()

    embeddings = {}
    for name, filename in (
        ("adaptive", "fusion_embeddings_adaptive.npy"),
        ("average", "fusion_embeddings_average.npy"),
        ("concatenation", "fusion_embeddings_concat.npy"),
    ):
        path = fusion_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} (expected from Step 11).")
        embeddings[name] = np.load(path)

    return subject_ids, embeddings


def find_participants_file(logger: logging.Logger) -> Path:
    for candidate in PARTICIPANTS_FILE_CANDIDATES:
        if candidate.exists():
            logger.info("Using participants file: %s", candidate)
            return candidate
    raise FileNotFoundError(
        f"Could not find a participants file in any of: "
        f"{[str(p) for p in PARTICIPANTS_FILE_CANDIDATES]}. "
        f"Add its actual path to PARTICIPANTS_FILE_CANDIDATES at the top of this file."
    )


def detect_column(columns: list[str], candidates: list[str], role: str, override: Optional[str], logger: logging.Logger) -> str:
    if override is not None:
        if override not in columns:
            raise ValueError(f"Override '{override}' for {role} not found in columns: {columns}")
        return override
    for candidate in candidates:
        if candidate in columns:
            logger.info("Auto-detected %s column: '%s'", role, candidate)
            return candidate
    raise ValueError(
        f"Could not auto-detect a {role} column among {candidates}. Available columns: {columns}. "
        f"Set the corresponding override at the top of this file."
    )


def load_labels(logger: logging.Logger) -> pd.DataFrame:
    participants_path = find_participants_file(logger)
    sep = "\t" if participants_path.suffix == ".tsv" else ","
    participants_df = pd.read_csv(participants_path, sep=sep)
    # Only the scanner/sex/age-MATCHED pairs. labels.tsv also lists extra
    # "lateralization_only" patients (for Step 13); including them here would
    # unbalance the classes and bring back the scanner confound (e.g. 104 HDx
    # patients vs 11 HDx controls).
    if "cohort" in participants_df.columns:
        participants_df = participants_df[participants_df["cohort"] == "matched"]
    participants_df = dev_only(participants_df)  # held-out test pairs stay locked away

    id_col = detect_column(list(participants_df.columns), PARTICIPANT_ID_COLUMN_CANDIDATES, "participant ID", None, logger)
    label_col = detect_column(list(participants_df.columns), LABEL_COLUMN_CANDIDATES, "label", LABEL_COLUMN_OVERRIDE, logger)

    raw_values = participants_df[label_col].astype(str).str.strip()
    logger.info("Raw label values found in '%s': %s", label_col, sorted(raw_values.unique().tolist()))

    positive_set = {v.lower() for v in POSITIVE_CLASS_VALUES}
    binary_label = raw_values.str.lower().isin(positive_set).astype(int)

    result = pd.DataFrame({
        "Subject_ID": participants_df[id_col].astype(str),
        "Label": binary_label,
        "Raw_Label": raw_values,
    })
    logger.info("Label distribution (1=positive/patient class): %s", result["Label"].value_counts().to_dict())
    return result


def align_labels(subject_ids: list[str], embeddings: dict[str, np.ndarray], labels_df: pd.DataFrame, logger: logging.Logger):
    label_lookup = dict(zip(labels_df["Subject_ID"], labels_df["Label"]))
    keep_idx = [i for i, sid in enumerate(subject_ids) if sid in label_lookup]

    n_dropped = len(subject_ids) - len(keep_idx)
    if n_dropped:
        logger.warning("%d/%d subjects have no matching label in participants file and were dropped.",
                        n_dropped, len(subject_ids))
    if not keep_idx:
        raise ValueError("No subjects matched between fusion_outputs/ and the participants file - "
                          "check Subject_ID formatting (e.g. 'sub-1' vs 'sub-01').")

    kept_ids = [subject_ids[i] for i in keep_idx]
    y = np.array([label_lookup[sid] for sid in kept_ids], dtype=int)
    aligned_embeddings = {name: arr[keep_idx] for name, arr in embeddings.items()}
    return kept_ids, aligned_embeddings, y


# ---------------------------------------------------------------------------
# LOOCV
# ---------------------------------------------------------------------------
def plain_loocv(X: np.ndarray, y: np.ndarray, estimator, seed: int) -> dict:
    n = len(y)
    loo = LeaveOneOut()
    y_true = np.zeros(n, dtype=int)
    y_pred = np.zeros(n, dtype=int)
    y_prob = np.zeros(n, dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        fitted = clone(estimator)
        if hasattr(fitted, "random_state"):
            fitted.set_params(random_state=seed)
        fitted.fit(X_train_s, y_train)

        y_true[test_idx] = y_test
        y_pred[test_idx] = fitted.predict(X_test_s)
        y_prob[test_idx] = fitted.predict_proba(X_test_s)[:, 1] if hasattr(fitted, "predict_proba") \
            else fitted.decision_function(X_test_s)

    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def pair_grouped_folds(groups: np.ndarray, n_splits: int, seed: int):
    """k folds of whole matched pairs, in a seed-dependent random order.
    Every pair is exactly 1 patient + 1 control, so each fold is balanced
    50/50 automatically. (sklearn's StratifiedGroupKFold was tried first:
    when all groups have the same class mix its greedy assignment ignores
    the shuffle seed and returns the SAME folds every repeat - 10 "repeats"
    with zero spread.)"""
    pairs = np.random.default_rng(seed).permutation(np.unique(groups))
    for test_pairs in np.array_split(pairs, n_splits):
        test_mask = np.isin(groups, test_pairs)
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


def repeated_grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, estimator,
                        n_splits: int, n_repeats: int, seed: int) -> pd.DataFrame:
    """Repeated stratified k-fold (k=5 -> 80/20 train/test per fold),
    grouped so each matched pair stays in one fold. Fixed (untuned)
    hyperparameters, scaler fit on the training fold only. One row per
    repeat: accuracy/AUC over all out-of-fold predictions of that repeat."""
    rows = []
    for r in range(n_repeats):
        y_prob = np.zeros(len(y))
        y_pred = np.zeros(len(y), dtype=int)
        for train_idx, test_idx in pair_grouped_folds(groups, n_splits, seed + r):
            scaler = StandardScaler().fit(X[train_idx])
            model = clone(estimator)
            if hasattr(model, "random_state"):
                model.set_params(random_state=seed + r)
            model.fit(scaler.transform(X[train_idx]), y[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_pred[test_idx] = model.predict(X_test)
            y_prob[test_idx] = model.predict_proba(X_test)[:, 1]
        rows.append({"repeat": r, "accuracy": accuracy_score(y, y_pred), "auc": roc_auc_score(y, y_prob)})
    return pd.DataFrame(rows)


def baseline_features(subject_ids: list[str]) -> dict[str, np.ndarray]:
    """Nuisance-only feature sets from labels.tsv. If the imaging model
    can't beat these, it hasn't learned anything beyond acquisition or
    demographics. With scanner/sex/age-matched pairs both should sit near
    chance (50%) - that's the check that the matching worked."""
    meta = pd.read_csv(LABELS_FILE, sep="\t").set_index("participant_id").loc[subject_ids]
    scanner = pd.get_dummies(meta["scanner"]).to_numpy(dtype=float)
    age_order = sorted(meta["age_band"].unique(), key=lambda b: (b != "<20", b))
    demographics = np.column_stack([
        (meta["sex"] == "M").astype(float),
        meta["age_band"].map({b: i for i, b in enumerate(age_order)}).astype(float),
    ])
    return {"scanner_only": scanner, "demographics_only": demographics}


def nested_loocv(X: np.ndarray, y: np.ndarray, estimator, param_grid: dict, inner_folds: int, seed: int) -> dict:
    n = len(y)
    loo = LeaveOneOut()
    y_true = np.zeros(n, dtype=int)
    y_pred = np.zeros(n, dtype=int)
    y_prob = np.zeros(n, dtype=float)
    best_params: list[dict] = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        if len(np.unique(y_train)) < 2:
            fitted = clone(estimator)
            if hasattr(fitted, "random_state"):
                fitted.set_params(random_state=seed)
            fitted.fit(X_train_s, y_train)
            best_params.append({})
        else:
            class_counts = np.bincount(y_train)
            inner_k = max(2, min(inner_folds, int(class_counts[class_counts > 0].min())))
            inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)

            base = clone(estimator)
            if hasattr(base, "random_state"):
                base.set_params(random_state=seed)

            try:
                search = GridSearchCV(base, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=1)
                search.fit(X_train_s, y_train)
                fitted = search.best_estimator_
                best_params.append(search.best_params_)
            except ValueError:
                fitted = clone(estimator)
                if hasattr(fitted, "random_state"):
                    fitted.set_params(random_state=seed)
                fitted.fit(X_train_s, y_train)
                best_params.append({})

        y_true[test_idx] = y_test
        y_pred[test_idx] = fitted.predict(X_test_s)
        y_prob[test_idx] = fitted.predict_proba(X_test_s)[:, 1] if hasattr(fitted, "predict_proba") \
            else fitted.decision_function(X_test_s)

    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "best_params": best_params}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> tuple[dict, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics, cm


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, aucs = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue

    def pct(values: list[float]) -> tuple[float, float]:
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (float("nan"), float("nan"))

    acc_lo, acc_hi = pct(accs)
    auc_lo, auc_hi = pct(aucs)
    return {"accuracy_ci_low": acc_lo, "accuracy_ci_high": acc_hi, "auc_ci_low": auc_lo, "auc_ci_high": auc_hi}


def permutation_test(X: np.ndarray, y: np.ndarray, estimator, observed_accuracy: float, n_permutations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    null_accuracies = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        result = plain_loocv(X, y_perm, estimator, seed=seed)
        null_accuracies.append(accuracy_score(result["y_true"], result["y_pred"]))
    null_accuracies = np.array(null_accuracies)
    p_value = float((np.sum(null_accuracies >= observed_accuracy) + 1) / (n_permutations + 1))
    return {"null_mean_accuracy": float(np.mean(null_accuracies)), "null_std_accuracy": float(np.std(null_accuracies)), "p_value": p_value}


def multi_seed_evaluation(X: np.ndarray, y: np.ndarray, estimator, n_seeds: int, logger: logging.Logger, model_name: str) -> pd.DataFrame:
    """Stability across random seeds (model init / solver randomness), at a
    FIXED hyperparameter setting - re-running the full nested hyperparameter
    search per seed would be prohibitively expensive for negligible extra
    insight, since hyperparameters were already selected once via the
    primary nested LOOCV above."""
    rows = []
    for seed in tqdm(range(n_seeds), desc=f"Multi-seed ({model_name})"):
        result = plain_loocv(X, y, estimator, seed=RANDOM_SEED + seed)
        metrics, _ = compute_classification_metrics(result["y_true"], result["y_pred"], result["y_prob"])
        rows.append({"seed": RANDOM_SEED + seed, "accuracy": metrics["accuracy"], "auc": metrics["auc"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_roc_curves(results_by_model: dict[str, dict], figures_dir: Path, logger: logging.Logger) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, result in results_by_model.items():
        try:
            fpr, tpr, _ = roc_curve(result["y_true"], result["y_prob"])
            auc_val = roc_auc_score(result["y_true"], result["y_prob"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
        except ValueError:
            continue
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (LOOCV, adaptive fusion embedding)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved roc_curve.png")


def plot_pr_curves(results_by_model: dict[str, dict], figures_dir: Path, logger: logging.Logger) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, result in results_by_model.items():
        precision, recall, _ = precision_recall_curve(result["y_true"], result["y_prob"])
        ax.plot(recall, precision, label=name)
    baseline = float(np.mean(next(iter(results_by_model.values()))["y_true"]))
    ax.axhline(baseline, linestyle="--", color="grey", label=f"Prevalence ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (LOOCV, adaptive fusion embedding)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "pr_curve.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved pr_curve.png")


def plot_confusion_matrices(results_by_model: dict[str, dict], figures_dir: Path, logger: logging.Logger) -> None:
    names = list(results_by_model.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = confusion_matrix(results_by_model[name]["y_true"], results_by_model[name]["y_pred"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Control", "Patient"])
        ax.set_yticklabels(["Control", "Patient"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved confusion_matrix.png")


def plot_calibration_curves(results_by_model: dict[str, dict], figures_dir: Path, logger: logging.Logger) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, result in results_by_model.items():
        n_bins = min(5, max(2, len(np.unique(result["y_prob"]))))
        try:
            prob_true, prob_pred = calibration_curve(result["y_true"], result["y_prob"], n_bins=n_bins, strategy="quantile")
            ax.plot(prob_pred, prob_true, marker="o", label=name)
        except ValueError:
            continue
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curves (LOOCV, adaptive fusion embedding)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "calibration_curve.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved calibration_curve.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for d in (CLASSIFIER_OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(LOGS_DIR / "step12.log")
    logger.info("Starting Step 12: Lightweight Classifier")

    t0 = time.perf_counter()
    subject_ids, embeddings = load_fusion_embeddings(FUSION_INPUT_DIR)
    labels_df = load_labels(logger)
    subject_ids, embeddings, y = align_labels(subject_ids, embeddings, labels_df, logger)

    if len(y) < 6 or len(np.unique(y)) < 2:
        logger.error("Not enough labeled subjects/classes (n=%d, classes=%s) to train classifiers.",
                      len(y), np.unique(y).tolist())
        return
    logger.info("Final dataset: %d subjects, class distribution=%s", len(y), np.bincount(y).tolist())
    t1 = time.perf_counter()

    X_primary = embeddings["adaptive"]

    # --- Primary: nested LOOCV per classifier on the adaptive-fusion embedding ---
    results_by_model: dict[str, dict] = {}
    metrics_rows: list[dict] = []
    confusion_matrices: dict[str, np.ndarray] = {}
    best_params_rows: list[dict] = []

    for name, spec in MODEL_SPECS.items():
        logger.info("Running nested LOOCV for %s ...", name)
        result = nested_loocv(X_primary, y, spec["estimator"], spec["param_grid"], INNER_CV_FOLDS, seed=RANDOM_SEED)
        results_by_model[name] = result
        metrics, cm = compute_classification_metrics(result["y_true"], result["y_pred"], result["y_prob"])
        confusion_matrices[name] = cm
        metrics["Model"] = name
        metrics_rows.append(metrics)
        for subject_id, params in zip(subject_ids, result["best_params"]):
            best_params_rows.append({"Model": name, "Subject_ID": subject_id, "Best_Params": str(params)})
        logger.info("%s LOOCV: accuracy=%.3f auc=%.3f", name, metrics["accuracy"], metrics["auc"])
    t2 = time.perf_counter()

    # --- Permutation testing (fixed, untuned estimator - nested search per
    # permutation would be computationally prohibitive) ---
    permutation_rows = []
    for name, spec in MODEL_SPECS.items():
        observed_acc = next(r for r in metrics_rows if r["Model"] == name)["accuracy"]
        logger.info("Running permutation test for %s (%d permutations) ...", name, N_PERMUTATIONS)
        perm_result = permutation_test(X_primary, y, spec["estimator"], observed_acc, N_PERMUTATIONS, seed=RANDOM_SEED)
        perm_result["Model"] = name
        perm_result["observed_accuracy"] = observed_acc
        permutation_rows.append(perm_result)
    t3 = time.perf_counter()

    # --- Multi-seed evaluation (repeats the full nested LOOCV) ---
    multi_seed_frames = []
    for name, spec in MODEL_SPECS.items():
        df = multi_seed_evaluation(X_primary, y, spec["estimator"], N_SEEDS, logger, name)
        df["Model"] = name
        multi_seed_frames.append(df)
    multi_seed_df = pd.concat(multi_seed_frames, ignore_index=True)
    t4 = time.perf_counter()

    # --- Bootstrap confidence intervals on the primary LOOCV predictions ---
    ci_rows = []
    for name, result in results_by_model.items():
        ci = bootstrap_ci(result["y_true"], result["y_pred"], result["y_prob"], N_BOOTSTRAP, seed=RANDOM_SEED)
        ci["Model"] = name
        ci_rows.append(ci)
    t5 = time.perf_counter()

    # --- Ablation / baseline comparison: untuned classifiers on the other
    # fusion methods, plus a majority-class dummy baseline ---
    ablation_rows = []
    for fusion_name, X_variant in embeddings.items():
        for model_name, spec in MODEL_SPECS.items():
            result = plain_loocv(X_variant, y, spec["estimator"], seed=RANDOM_SEED)
            metrics, _ = compute_classification_metrics(result["y_true"], result["y_pred"], result["y_prob"])
            ablation_rows.append({"Fusion_Method": fusion_name, "Model": model_name,
                                   "Accuracy": metrics["accuracy"], "AUC": metrics["auc"], "F1": metrics["f1"]})

    # (The majority-class dummy moved to the repeated grouped CV below - see
    # the note there on why LOOCV reports 0% for it on balanced classes.)

    # Repeated grouped 5-fold (80/20) CV for every fusion method x model,
    # plus the nuisance baselines. NOTE: baselines are evaluated HERE, not
    # with LOOCV. With exactly 50/50 classes, leaving one subject out makes
    # the OTHER class the training majority, so any model that has learned
    # nothing predicts the wrong class every time and LOOCV reports 0%
    # (the dummy/scanner/demographics rows in ablation.csv). Pair-grouped
    # folds stay 50/50, so an uninformative baseline correctly lands ~50%.
    groups = pd.read_csv(LABELS_FILE, sep="\t").set_index("participant_id").loc[subject_ids, "pair"].to_numpy()
    repeated_frames = []
    for fusion_name, X_variant in embeddings.items():
        for model_name, spec in MODEL_SPECS.items():
            df = repeated_grouped_cv(X_variant, y, groups, spec["estimator"], CV_N_SPLITS, CV_N_REPEATS, RANDOM_SEED)
            repeated_frames.append(df.assign(Fusion_Method=fusion_name, Model=model_name))
    for baseline_name, X_base in baseline_features(subject_ids).items():
        df = repeated_grouped_cv(X_base, y, groups, LogisticRegression(max_iter=2000), CV_N_SPLITS, CV_N_REPEATS, RANDOM_SEED)
        repeated_frames.append(df.assign(Fusion_Method=baseline_name, Model="logistic_regression"))
    df = repeated_grouped_cv(X_primary, y, groups, DummyClassifier(strategy="prior"), CV_N_SPLITS, CV_N_REPEATS, RANDOM_SEED)
    repeated_frames.append(df.assign(Fusion_Method="none", Model="dummy_prior"))
    repeated_df = pd.concat(repeated_frames, ignore_index=True)
    repeated_df.to_csv(METRICS_DIR / "repeated_cv_per_repeat.csv", index=False)
    repeated_summary = repeated_df.groupby(["Fusion_Method", "Model"]).agg(
        mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
        mean_auc=("auc", "mean"), std_auc=("auc", "std"),
    ).reset_index()
    repeated_summary.to_csv(METRICS_DIR / "repeated_cv_summary.csv", index=False)
    for row in repeated_summary.itertuples():
        logger.info("Repeated %dx%d-fold CV %s/%s: accuracy=%.3f+-%.3f auc=%.3f+-%.3f",
                    CV_N_REPEATS, CV_N_SPLITS, row.Fusion_Method, row.Model,
                    row.mean_accuracy, row.std_accuracy, row.mean_auc, row.std_auc)
    t6 = time.perf_counter()

    # --- Final deployable models: refit each classifier on all data ---
    final_models = {}
    scaler_final = StandardScaler().fit(X_primary)
    X_primary_scaled = scaler_final.transform(X_primary)
    for name, spec in MODEL_SPECS.items():
        n_classes_ok = len(np.unique(y)) > 1
        cv_k = max(2, min(INNER_CV_FOLDS, int(np.bincount(y).min()))) if n_classes_ok else 2
        try:
            search = GridSearchCV(clone(spec["estimator"]), spec["param_grid"],
                                   cv=StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=RANDOM_SEED),
                                   scoring="accuracy", n_jobs=1)
            search.fit(X_primary_scaled, y)
            final_models[name] = search.best_estimator_
        except ValueError:
            fallback = clone(spec["estimator"])
            fallback.fit(X_primary_scaled, y)
            final_models[name] = fallback
    t7 = time.perf_counter()

    # --- Save classifier_outputs/ ---
    predictions_df = pd.DataFrame({"Subject_ID": subject_ids, "y_true": y})
    for name, result in results_by_model.items():
        predictions_df[f"{name}_pred"] = result["y_pred"]
        predictions_df[f"{name}_prob"] = result["y_prob"]
    predictions_df.to_csv(CLASSIFIER_OUTPUT_DIR / "predictions.csv", index=False)
    pd.DataFrame(best_params_rows).to_csv(CLASSIFIER_OUTPUT_DIR / "best_hyperparameters.csv", index=False)
    logger.info("Saved predictions -> %s", CLASSIFIER_OUTPUT_DIR)

    for name, model in final_models.items():
        with (MODELS_DIR / f"step12_{name}.pkl").open("wb") as f:
            pickle.dump({"model": model, "scaler": scaler_final}, f)
    logger.info("Saved final refit models -> %s", MODELS_DIR)

    # --- Metrics ---
    pd.DataFrame(metrics_rows)[["Model", "accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "auc"]].to_csv(
        METRICS_DIR / "statistics.csv", index=False
    )

    validation_df = pd.DataFrame(permutation_rows).merge(pd.DataFrame(ci_rows), on="Model")
    seed_summary = multi_seed_df.groupby("Model").agg(
        mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
        mean_auc=("auc", "mean"), std_auc=("auc", "std"),
    ).reset_index()
    validation_df = validation_df.merge(seed_summary, on="Model")
    validation_df.to_csv(METRICS_DIR / "validation.csv", index=False)

    pd.DataFrame(ablation_rows).to_csv(METRICS_DIR / "ablation.csv", index=False)

    best_row = max(metrics_rows, key=lambda r: r["auc"] if not np.isnan(r["auc"]) else -1)
    pd.DataFrame([{
        "n_subjects": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "best_model": best_row["Model"],
        "best_model_accuracy": best_row["accuracy"],
        "best_model_auc": best_row["auc"],
    }]).to_csv(METRICS_DIR / "summary.csv", index=False)

    pd.DataFrame([{
        "load_data_seconds": round(t1 - t0, 3),
        "nested_loocv_seconds": round(t2 - t1, 3),
        "permutation_test_seconds": round(t3 - t2, 3),
        "multi_seed_seconds": round(t4 - t3, 3),
        "bootstrap_ci_seconds": round(t5 - t4, 3),
        "ablation_seconds": round(t6 - t5, 3),
        "final_refit_seconds": round(t7 - t6, 3),
        "total_seconds": round(t7 - t0, 3),
    }]).to_csv(METRICS_DIR / "runtime.csv", index=False)

    # --- Visualizations ---
    plot_roc_curves(results_by_model, FIGURES_DIR, logger)
    plot_pr_curves(results_by_model, FIGURES_DIR, logger)
    plot_confusion_matrices(results_by_model, FIGURES_DIR, logger)
    plot_calibration_curves(results_by_model, FIGURES_DIR, logger)

    logger.info(
        "Step 12 complete. Best model: %s (accuracy=%.3f, AUC=%.3f) | Total runtime=%.2f s",
        best_row["Model"], best_row["accuracy"], best_row["auc"], t7 - t0,
    )


if __name__ == "__main__":
    main()
