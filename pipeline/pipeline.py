"""
pipeline.py — Assignment 1: Build the Foundation

You are building the data pipeline for a clinical machine learning task:
predicting which patients are likely to have chronic pain, using a synthetic
dataset of 320 patient records.

Your job is to implement 8 functions — no external ML libraries, just NumPy.
Two helper functions (standardize_features, predict_proba) are already
implemented for you; do not change them.

─────────────────────────────────────────────────────────────────────────────
FUNCTIONS YOU MUST IMPLEMENT
─────────────────────────────────────────────────────────────────────────────

  1. flag_keyword_match       Turn raw condition text into a binary pain label
                              by scanning for pain-related keywords.

  2. stratified_split         Split the dataset into train and validation sets
                              while preserving the class ratio in both halves.

  3. sigmoid                  The logistic activation: maps any real number to
                              a probability in (0, 1).

  4. logistic_regression_     Compute the gradient of the log-loss so the
     gradients                training loop knows which direction to move.

  5. train_logistic_          Run gradient descent for a fixed number of steps,
     regression               returning the trained weights and bias.

  6. confusion_counts         Count TP, FP, TN, FN between predicted and true
                              binary labels.

  7. precision_recall_f1      Compute precision, recall, and F1 from the
                              confusion counts.

  8. roc_auc                  Compute the AUC using pairwise comparison of
                              every (positive, negative) score pair.

─────────────────────────────────────────────────────────────────────────────
WORKFLOW
─────────────────────────────────────────────────────────────────────────────

  1. Complete the assignment-1 ui exploration to find your optimal hyperparameters.
  2. Fill in get_sandbox_params() with your NetID and hyperparameters from the UI.
  3. Implement all 8 functions below.
  4. Run:  'python pipeline.py'   to verify your results match the oracle.
  5. Run:  'pytest test_pipeline.py -v'   to verify your implementation passes all tests.
  6. Commit pipeline.py and your submission file, then push.

Do not modify standardize_features, predict_proba, or anything below the
'Local verification' section.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ======================================================================================
# Assignment-1 UI parameters  ← fill in after completing the assignment-1 UI exploration
# ======================================================================================

def get_sandbox_params() -> dict:
    """Return your student ID and the hyperparameters you discovered in the
    assignment-1 UI exploration that reached optimal performance for your personal seed.

    After the Training Explorer shows the green 'Optimal Performance Reached!' banner, 
    copy those three values here alongside your NetID.

    Your student_id is used to derive the same seed.

    :return: dict with keys 'student_id' (str), 'learning_rate' (float),
             'steps' (int), 'val_fraction' (float).
    """
    params = {
            "student_id": "yanghanfei",
            "learning_rate": 1.000,
            "steps": 500,
            "val_fraction": 0.20,
    }

    if params["val_fraction"] != 0.20:
        raise ValueError(
            "val_fraction must be exactly 0.20 -- do not change this value, "
            "the oracle always uses a 20% validation split."
        )
    
    return params


# ---------------------------------------------------------------------
# Do not change this list. This is not perfectly comprehensive, but
# it is a reasonable set of keywords to flag patients with chronic pain.
# ---------------------------------------------------------------------

PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
]


def flag_keyword_match(descriptions: List[str], keywords: List[str]) -> np.ndarray:

    pattern = r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
    matches = [bool(re.search(pattern, desc.lower())) for desc in descriptions] 
    return np.array(matches, dtype=bool)



def standardize_features(
    feature_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PROVIDED — you do not need to edit this function.

    Z-scores each column of feature_matrix (subtract the column mean,
    divide by the column standard deviation) so that gradient descent in
    train_logistic_regression converges in a reasonable number of
    iterations regardless of each feature's original scale (age in years
    vs. healthcare expenses in dollars, for example).

    :param feature_matrix: A 2-D array of shape (N, D).
    :return: A tuple (scaled_matrix, means, stds).
    """
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return (feature_matrix - means) / stds, means, stds


def stratified_split(labels: np.ndarray, val_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(labels), dtype=bool)
    unique_classes = np.unique(labels)
    
    for c in unique_classes:

        class_indices = np.where(labels == c)[0]
        shuffled_indices = rng.permutation(class_indices)
        
        n_val = round(val_fraction * len(class_indices))
        val_mask[shuffled_indices[:n_val]] = True
        
    return val_mask


def sigmoid(z: np.ndarray) -> np.ndarray:

    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def predict_proba(
    feature_matrix: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    """PROVIDED — you do not need to edit this function.

    Computes the predicted probability of the positive class for each row
    of feature_matrix, using your sigmoid() function.

    :param feature_matrix: A 2-D array of shape (N, D).
    :param weights: A length-D array.
    :param bias: A scalar.
    :return: A length-N array of probabilities in (0, 1).
    """
    return sigmoid(feature_matrix.dot(weights) + bias)


def logistic_regression_gradients(
    feature_matrix: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> Tuple[np.ndarray, float]:
    n_samples = feature_matrix.shape[0]

    predictions = predict_proba(feature_matrix, weights, bias)
    error = predictions - y
    grad_weights = (feature_matrix.T.dot(error)) / n_samples
    grad_bias = np.mean(error)
    
    return grad_weights, grad_bias


def train_logistic_regression(
    feature_matrix: np.ndarray,
    y: np.ndarray,
    iterations: int,
    learning_rate: float,
) -> Tuple[np.ndarray, float]:
    n_features = feature_matrix.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0
    
    for _ in range(iterations):
        grad_w, grad_b = logistic_regression_gradients(feature_matrix, y, weights, bias)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        
    return weights, bias


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    
    correct_pairs = 0.0
    total_pairs = len(pos_scores) * len(neg_scores)
    
    if total_pairs == 0:
        return 0.0
    for p_score in pos_scores:
        for n_score in neg_scores:
            if p_score > n_score:
                correct_pairs += 1.0
            elif p_score == n_score:
                correct_pairs += 0.5 
                
    return correct_pairs / total_pairs


# =============================================================================
#
# Local verification  ─  run:  python pipeline.py
#
# Do not modify anything below this line.
#
# =============================================================================

_FEATURE_COLS = [
    "age", "is_female", "number_of_unique_meds", "number_of_encounters",
    "number_of_procedures", "unique_procedures", "pain_severity",
    "body_height", "body_weight", "body_mass_index",
    "systolic_blood_pressure", "diastolic_blood_pressure",
    "heart_rate", "respiratory_rate",
    "qaly", "daly", "qols", "healthcare_expenses", "healthcare_coverage",
]

def _run_metrics(data_path, seed, lr, steps, vf): 
    """Run the full pipeline and return a metrics dict. Not part of the graded API."""
    df = pd.read_csv(data_path)
    labels = flag_keyword_match(df["condition_text"].tolist(), PAIN_KEYWORDS).astype(float)
    x_all = df[_FEATURE_COLS].values.astype(float)

    is_val = stratified_split(labels.astype(int), vf, seed)
    x_train, y_train = x_all[~is_val], labels[~is_val]
    x_val, y_val     = x_all[is_val],  labels[is_val]

    x_train_s, means, stds = standardize_features(x_train)
    x_val_s = (x_val - means) / stds

    weights, bias = train_logistic_regression(
        x_train_s, y_train, iterations=int(steps), learning_rate=float(lr)
    )
    probs  = predict_proba(x_val_s, weights, bias)
    y_pred = (probs >= 0.5).astype(int)

    tp_v, fp_v, tn_v, fn_v = confusion_counts(y_val.astype(int), y_pred)
    _, _, f1_v = precision_recall_f1(tp_v, fp_v, fn_v)
    auc_v = roc_auc(y_val.astype(int), probs)
    acc_v = (tp_v + tn_v) / len(y_val)
    train_probs = predict_proba(x_train_s, weights, bias)
    eps = 1e-9
    loss_v = float(-np.mean(
        y_train * np.log(train_probs + eps) + (1 - y_train) * np.log(1 - train_probs + eps)
    ))
    return {"auc": auc_v, "accuracy": acc_v, "f1": f1_v, "loss": loss_v}


def _verify(): 
    """Local verification — called by __main__. Not part of the graded API."""
    root = Path(__file__).parent.parent   # assignment-1/

    p = get_sandbox_params()
    student_id = p.get("student_id", "").strip()
    lr, steps, vf = p["learning_rate"], p["steps"], p["val_fraction"]
    if not student_id:
        print("\n  Fill in 'student_id' in get_sandbox_params() with your NetID.\n")
        raise SystemExit(1)
    if lr == 0 or steps == 0:
        print("\n  Fill in 'learning_rate' and 'steps' in get_sandbox_params()\n"
              "  with the values from the UI Training Explorer.\n")
        raise SystemExit(1)

    # Derive the same seed used by the UI explorer (deterministic from student_id)
    h_val = int(hashlib.sha256(student_id.lower().encode()).hexdigest(), 16)
    seed = h_val % 900 + 100

    print(f"\n  Student : {student_id}   Seed: {seed}")
    print(f"  Params  : lr={lr}, steps={steps}, val_fraction={vf}\n")

    m = _run_metrics(root / "data" / "patient_features.csv", seed, lr, steps, vf)

    print("  Metric    Value")
    print("  ------------------")
    print(f"  {'F1':<9} {m['f1']:.3f}")
    print(f"  {'Accuracy':<9} {m['accuracy']:.3f}")
    print(f"  {'AUC':<9} {m['auc']:.3f}")
    print(f"  {'Loss':<9} {m['loss']:.3f}")
    print()
    print("  Note: This local output is informational only and does not reveal hidden test code.")
    print("  The official grader will perform the final evaluation.")
    print("  Run 'pytest test_pipeline.py -v' to verify your implementation passes all tests.")
    print("  Submit your pipeline.py to GitHub for grading.")


if __name__ == "__main__":
    _verify()

