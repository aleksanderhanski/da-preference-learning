"""Shared helpers for the Cars preference-learning models.

Used by xgboost_cars.py, ann_utadis_cars.py and deep_nn_cars.py so that the
dataset loading, class construction, metric reporting, alternative selection,
model-agnostic min-flip search and permutation feature importance stay in one
place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


FEATURE_NAMES = ['HorsePower', 'Cars Prices', 'Seats', 'Total Speed']

CRITERION_DIRECTION = {
    'HorsePower':  +1,
    'Cars Prices': -1,
    'Seats':       +1,
    'Total Speed': +1,
}

CRITERION_UNIT = {
    'HorsePower':  'hp',
    'Cars Prices': 'USD',
    'Seats':       'count',
    'Total Speed': 'km/h',
}


def _normalise(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())


def load_data(csv_path: str = "dataset/dataset_preprocessed_continuous.csv",
              test_size: float = 0.2, random_state: int = 1234):
    """Load the preprocessed Cars dataset and build the binary target.

    Returns
    -------
    df_raw        : original dataframe (for car names)
    df            : dataframe with utility + class columns
    X, y          : feature frame and target series
    X_train/test  : 80/20 split
    y_train/test  : matching target split
    feature_ranges: dict feat -> (min, max) from the full dataset
    """
    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()
    df['utility'] = (
        _normalise(df['HorsePower']) +
        _normalise(df['Cars Prices'].max() - df['Cars Prices']) +
        _normalise(df['Seats']) +
        _normalise(df['Total Speed'])
    )
    n = len(df)
    df['class'] = (df['utility'].rank(method='first', ascending=True) > n // 2).astype(int)

    X = df[FEATURE_NAMES].astype(float).copy()
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    feature_ranges = {f: (float(df[f].min()), float(df[f].max())) for f in FEATURE_NAMES}
    return df_raw, df, X, y, X_train, X_test, y_train, y_test, feature_ranges


def report_metrics(predict_fn, predict_proba_fn,
                   X_train, y_train, X_test, y_test) -> dict:
    """Print Accuracy / F1 / AUC on train+test rounded to 4 dp.

    `predict_fn` and `predict_proba_fn` must accept a pandas DataFrame and
    return numpy arrays (class labels and positive-class probabilities).
    Returns the metrics dict for programmatic use.
    """
    out = {}
    for name, X_s, y_s in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        y_pred = np.asarray(predict_fn(X_s))
        y_prob = np.asarray(predict_proba_fn(X_s))
        acc = accuracy_score(y_s, y_pred)
        f1  = f1_score(y_s, y_pred)
        auc = roc_auc_score(y_s, y_prob)
        out[name] = {"Accuracy": round(acc, 4), "F1": round(f1, 4), "AUC": round(auc, 4)}
        print(f"{name:5s} | Accuracy={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return out


def select_three_alternatives(df_pred: pd.DataFrame):
    """Pick a preferred / borderline / not-preferred alternative by pred_prob.

    `df_pred` must contain a `pred_prob` column with the model's probabilities.
    Returns [(idx, label), ...] ready to iterate.
    """
    high_idx = df_pred['pred_prob'].idxmax()
    low_idx  = df_pred['pred_prob'].idxmin()
    median_prob = df_pred['pred_prob'].median()
    mid_idx  = (df_pred['pred_prob'] - median_prob).abs().idxmin()
    return [
        (high_idx, 'Preferred'),
        (mid_idx,  'Borderline'),
        (low_idx,  'Not preferred'),
    ]


def get_name(df_raw: pd.DataFrame, idx) -> str:
    row = df_raw.loc[idx]
    return f"{row['Company Names']} {row['Cars Names']}"


def find_min_flip_sampling(predict_fn, row, feature_names, feature_ranges,
                           grid: int = 400):
    """Grid-search the minimum single-criterion change that flips the class.

    Model-agnostic: calls `predict_fn(DataFrame) -> array of 0/1`.
    For each criterion we build a dense grid of candidate values on [lo, hi],
    predict in one batched call, then pick the smallest |delta| that flips.
    Returns a list of dicts (one per criterion where a flip was found).
    """
    row_vals = np.asarray(row[feature_names].values, dtype=float)
    X_row = pd.DataFrame([row_vals], columns=feature_names)
    current_pred = int(np.asarray(predict_fn(X_row))[0])
    results = []

    for feat in feature_names:
        orig = float(row[feat])
        lo, hi = feature_ranges[feat]
        values = np.linspace(lo, hi, grid)

        base = np.tile(row_vals, (grid, 1))
        batch = pd.DataFrame(base, columns=feature_names)
        batch[feat] = values

        preds = np.asarray(predict_fn(batch))
        flipped = preds != current_pred
        if not flipped.any():
            continue

        deltas = values - orig
        abs_deltas = np.abs(deltas).astype(float)
        abs_deltas[~flipped] = np.inf
        i = int(np.argmin(abs_deltas))
        results.append({
            'criterion': feat,
            'orig_value': orig,
            'new_value': float(values[i]),
            'delta': float(deltas[i]),
            'old_pred': current_pred,
            'new_pred': int(preds[i]),
        })
    return results


def permutation_feature_importance(predict_proba_fn, X: pd.DataFrame, y: pd.Series,
                                   feature_names=None, n_repeats: int = 30,
                                   random_state: int = 0) -> dict:
    """AUC-drop permutation importance, usable with any probabilistic model.

    Returns {feature: (mean_drop, std_drop)} averaged over `n_repeats` shuffles.
    """
    if feature_names is None:
        feature_names = list(X.columns)
    rng = np.random.default_rng(random_state)
    baseline = roc_auc_score(y, np.asarray(predict_proba_fn(X)))
    out = {}
    for feat in feature_names:
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].values)
            auc_perm = roc_auc_score(y, np.asarray(predict_proba_fn(X_perm)))
            drops.append(baseline - auc_perm)
        out[feat] = (float(np.mean(drops)), float(np.std(drops)))
    return out


def plot_permutation_importance(importances: dict,
                                title: str = "Permutation Feature Importance"):
    feats = list(importances.keys())
    means = [importances[f][0] for f in feats]
    stds  = [importances[f][1] for f in feats]
    order = np.argsort(means)
    feats_sorted = [feats[i] for i in order]
    means_sorted = [means[i] for i in order]
    stds_sorted  = [stds[i]  for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(feats_sorted, means_sorted, xerr=stds_sorted, color='#4a90d9')
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_xlabel("Mean AUC drop when criterion is shuffled (higher = more important)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def manual_pdp(predict_proba_fn, X_background: pd.DataFrame, feature: str,
               feature_range, grid: int = 60):
    """Compute 1-D PDP for any predict-proba callable. Returns (xs, mean_probs)."""
    lo, hi = feature_range
    xs = np.linspace(lo, hi, grid)
    probs = np.empty(grid)
    base = X_background.values.astype(float)
    n = len(base)
    for i, v in enumerate(xs):
        batch = base.copy()
        batch[:, list(X_background.columns).index(feature)] = v
        probs[i] = float(np.mean(predict_proba_fn(pd.DataFrame(batch, columns=X_background.columns))))
    return xs, probs


def manual_ice(predict_proba_fn, X_background: pd.DataFrame, feature: str,
               feature_range, grid: int = 60):
    """Compute ICE curves for any predict-proba callable.

    Returns (xs, curves) where curves has shape (n_samples, grid).
    """
    lo, hi = feature_range
    xs = np.linspace(lo, hi, grid)
    base = X_background.values.astype(float)
    n = len(base)
    col_idx = list(X_background.columns).index(feature)
    curves = np.empty((n, grid))
    for i, v in enumerate(xs):
        batch = base.copy()
        batch[:, col_idx] = v
        curves[:, i] = np.asarray(predict_proba_fn(pd.DataFrame(batch, columns=X_background.columns)))
    return xs, curves


def plot_pdp_ice_grid(predict_proba_fn, X_background, feature_names, feature_ranges,
                      title_prefix: str = ""):
    """Plot 1-D PDP + ICE for each criterion in a grid."""
    n = len(feature_names)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
    for j, f in enumerate(feature_names):
        xs, probs = manual_pdp(predict_proba_fn, X_background, f, feature_ranges[f])
        axes[0, j].plot(xs, probs, color='#d9534f', lw=2)
        axes[0, j].set_title(f"PDP — {f}")
        axes[0, j].set_xlabel(f)
        axes[0, j].set_ylabel("mean P(class=1)")

        xs, curves = manual_ice(predict_proba_fn, X_background, f, feature_ranges[f])
        for row in curves:
            axes[1, j].plot(xs, row, color='#4a90d9', alpha=0.25, lw=0.8)
        axes[1, j].plot(xs, curves.mean(0), color='#d9534f', lw=2)
        axes[1, j].set_title(f"ICE — {f}")
        axes[1, j].set_xlabel(f)
        axes[1, j].set_ylabel("P(class=1)")
    if title_prefix:
        plt.suptitle(title_prefix, y=1.02)
    plt.tight_layout()
    plt.show()
