# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: myenv
#     language: python
#     name: myenv
# ---

# %% [markdown]
# # Neural Network MLP — Car Preference Learning
#
# **Aleksander Hański 160315 and Michał Żurawski 160252**
#
# We use the preprocessed Cars 2025 dataset (177 alternatives, 4 continuous criteria)
# to train a multi-layer perceptron (MLP) with ReLU activations that models Michał's preferences.
# Unlike XGBoost, the MLP can learn arbitrary non-linear interactions between criteria.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from utils import (
    NumpyDataset, CreateDataLoader, Regret, Accuracy, AUC, ScoreTracker,
)

torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## 1. Load data

# %%
df_raw = pd.read_csv("dataset/dataset_preprocessed_continuous.csv")
print(df_raw.shape)
df_raw.head()

# %% [markdown]
# ## 2. Criteria and target class
#
# Same utility construction as the XGBoost model so results are directly comparable.
#
# | Criterion   | Type | Unit  | Direction   |
# |-------------|------|-------|-------------|
# | HorsePower  | gain | hp    | higher → better |
# | Cars Prices | cost | USD   | lower  → better |
# | Seats       | gain | count | higher → better |
# | Total Speed | gain | km/h  | higher → better |

# %%
df = df_raw.copy()
FEATURE_NAMES = ['HorsePower', 'Cars Prices', 'Seats', 'Total Speed']


def normalise(s):
    return (s - s.min()) / (s.max() - s.min())


df['utility'] = (
    normalise(df['HorsePower']) +
    normalise(df['Cars Prices'].max() - df['Cars Prices']) +   # inverted: cheaper = better
    normalise(df['Seats']) +
    normalise(df['Total Speed'])
)

n = len(df)
df['class'] = (df['utility'].rank(method='first', ascending=True) > n // 2).astype(int)

print("Class distribution:")
print(df['class'].value_counts().rename({0: 'class 0 (not preferred)', 1: 'class 1 (preferred)'}))

# %%
X_all = df[FEATURE_NAMES].astype(float).values
y_all = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=1234
)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

FEATURE_RANGES = {f: (df[f].min(), df[f].max()) for f in FEATURE_NAMES}

# %% [markdown]
# ## 3. Feature normalisation to [0, 1]
#
# Neural networks are sensitive to input scale.  We normalise each criterion to [0, 1]
# using min/max computed on the **training set** only.

# %%
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)


def scale(arr: np.ndarray) -> np.ndarray:
    return (arr - X_min) / (X_max - X_min + 1e-8)


X_train_s = scale(X_train)
X_test_s  = scale(X_test)
X_all_s   = scale(X_all)

# %% [markdown]
# ## 4. DataLoaders

# %%
train_loader = CreateDataLoader(X_train_s, y_train)
test_loader  = CreateDataLoader(X_test_s, y_test)

# %% [markdown]
# ## 5. MLP model definition
#
# Three hidden layers with ReLU activations followed by a linear output layer.
# The output is an **unbounded scalar** (no sigmoid):
# - positive output → predicted class 1 (preferred)
# - negative output → predicted class 0 (not preferred)
#
# This is compatible with the `Regret` loss used in `utils.py`.

# %%
class MLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dims: tuple = (32, 16, 8)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # shape: (batch,)


model = MLP()
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# %% [markdown]
# ## 6. Training (AdamW + OneCycleLR, 300 epochs)
#
# `Train` from `utils.py` handles the loop: AdamW optimiser with OneCycleLR scheduler,
# Regret loss, and saves the checkpoint with the best training accuracy.

# %%
CHECKPOINT = "mlp_cars.pt"
LR         = 0.01
EPOCHS     = 300


def train_mlp(
    model:          nn.Module,
    train_loader,
    test_loader,
    path:           str,
    lr:             float = 0.01,
    epoch_nr:       int   = 300,
):
    """Training loop: AdamW + OneCycleLR, Regret loss, saves best checkpoint."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epoch_nr
    )
    best_acc   = 0.0
    best_auc   = 0.0
    stats_train = ScoreTracker()
    stats_test  = ScoreTracker()

    from tqdm import tqdm
    for epoch in tqdm(range(epoch_nr)):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = Regret(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            acc = float(Accuracy(outputs, labels))
            auc = float(AUC(outputs, labels))
            stats_train.append(loss.item(), auc, acc)

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                stats_test.append(
                    Regret(outputs, labels).item(),
                    float(AUC(outputs, labels)),
                    float(Accuracy(outputs, labels)),
                )

        if acc > best_acc:
            best_acc = acc
            best_auc = auc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict()},
                path,
            )

    return best_acc, best_auc, stats_train, stats_test


best_acc, best_auc, stats_train, stats_test = train_mlp(
    model, train_loader, test_loader, path=CHECKPOINT, lr=LR, epoch_nr=EPOCHS,
)

# Load the best checkpoint
ckpt = torch.load(CHECKPOINT, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"\nBest train  Acc={best_acc:.4f}  AUC={best_auc:.4f}  (saved at epoch {ckpt['epoch']})")

# %% [markdown]
# ## 7. Training curves

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(stats_train.losses,    label='train')
axes[0].plot(stats_test.losses,     label='test')
axes[0].set_title("Regret loss")
axes[0].legend()

axes[1].plot(stats_train.acc_scores, label='train')
axes[1].plot(stats_test.acc_scores,  label='test')
axes[1].set_title("Accuracy")
axes[1].legend()

axes[2].plot(stats_train.auc_scores, label='train')
axes[2].plot(stats_test.auc_scores,  label='test')
axes[2].set_title("AUC")
axes[2].legend()

plt.suptitle("MLP Training Curves")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Metrics: Accuracy, F1, AUC

# %%
def predict_prob(arr_scaled: np.ndarray) -> np.ndarray:
    """Return P(class=1) via sigmoid on the model's raw output."""
    t = torch.tensor(arr_scaled, dtype=torch.float32)
    with torch.no_grad():
        raw = model(t).numpy()
    return torch.sigmoid(torch.tensor(raw)).numpy()


def predict_class(arr_scaled: np.ndarray) -> np.ndarray:
    """Return hard class label (threshold at raw output = 0)."""
    t = torch.tensor(arr_scaled, dtype=torch.float32)
    with torch.no_grad():
        raw = model(t).numpy()
    return (raw > 0).astype(int)


for split_name, Xs, ys in [("Train", X_train_s, y_train), ("Test", X_test_s, y_test)]:
    y_pred = predict_class(Xs)
    y_prob = predict_prob(Xs)
    acc = accuracy_score(ys, y_pred)
    f1  = f1_score(ys, y_pred)
    auc = roc_auc_score(ys, y_prob)
    print(f"{split_name:5s} | Accuracy={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

# %% [markdown]
# ## 9. Select 3 alternatives for deep explanation
#
# We pick the most preferred, most borderline, and least preferred alternative
# (same selection strategy as the XGBoost model).

# %%
df_pred = df.copy()
df_pred['pred']      = predict_class(X_all_s)
df_pred['pred_prob'] = predict_prob(X_all_s)


def get_name(idx: int) -> str:
    row = df_raw.loc[idx]
    return f"{row['Company Names']} {row['Cars Names']}"


high_idx    = df_pred['pred_prob'].idxmax()
low_idx     = df_pred['pred_prob'].idxmin()
median_prob = df_pred['pred_prob'].median()
mid_idx     = (df_pred['pred_prob'] - median_prob).abs().idxmin()

selected = [high_idx, mid_idx, low_idx]
labels   = ['Preferred', 'Borderline', 'Not preferred']

print("Selected alternatives:")
for idx, lbl in zip(selected, labels):
    row = df_pred.loc[idx]
    print(f"  [{lbl}] {get_name(idx)}")
    print(f"    Criteria: HP={row['HorsePower']:.0f} hp, Price=${row['Cars Prices']:.0f}, "
          f"Seats={row['Seats']:.0f}, Speed={row['Total Speed']:.0f} km/h")
    print(f"    Utility={row['utility']:.4f}  Pred prob={row['pred_prob']:.4f}  "
          f"Class={int(row['pred'])}")

# %% [markdown]
# ## 10. SHAP explanations
#
# We use `shap.GradientExplainer` (expected gradients) — the method suited for
# differentiable PyTorch models.  The training set is used as background.
# SHAP values are in logit space; sign and magnitude are what matter.

# %%
# GradientExplainer requires the model to return 2D output (batch, outputs)
class ShapWrapper(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return model(x).unsqueeze(-1)   # (batch,) → (batch, 1)

shap_model  = ShapWrapper()
shap_model.eval()

background       = torch.tensor(X_train_s, dtype=torch.float32)
explainer        = shap.GradientExplainer(shap_model, background)

X_all_t          = torch.tensor(X_all_s, dtype=torch.float32)
shap_values_raw  = explainer.shap_values(X_all_t)

# GradientExplainer returns shape (n_samples, n_features, n_outputs)
shap_values  = shap_values_raw[:, :, 0]       # shape (n, 4)

# GradientExplainer has no expected_value attribute — compute it as the mean
# model output over the background (= E[f(X)] in logit space)
with torch.no_grad():
    expected_val = float(model(background).mean().item())

print(f"SHAP values shape: {shap_values.shape}")
print(f"Expected value (baseline logit): {expected_val:.4f}")

# %%
for idx, lbl in zip(selected, labels):
    pos = df_pred.index.get_loc(idx)
    explanation = shap.Explanation(
        values        = shap_values[pos],
        base_values   = expected_val,
        data          = X_all[pos],           # original scale for readability
        feature_names = FEATURE_NAMES,
    )
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP — {lbl}: {get_name(idx)}", pad=12)
    plt.tight_layout()
    plt.show()

# %%
shap.summary_plot(shap_values, X_all_s, feature_names=FEATURE_NAMES, show=False)
plt.title("SHAP Summary (beeswarm)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Partial Dependence Plots (PDP) and ICE plots
#
# We wrap the PyTorch model in a scikit-learn compatible estimator so that
# `PartialDependenceDisplay` can call it with arbitrary feature values.
# The wrapper applies the same [0,1] normalisation internally.

# %%
class SklearnWrapper(ClassifierMixin, BaseEstimator):
    """Thin sklearn wrapper around the trained MLP."""

    def __init__(self, torch_model: nn.Module, x_min: np.ndarray, x_max: np.ndarray):
        self.torch_model = torch_model
        self.x_min       = x_min
        self.x_max       = x_max
        self.classes_    = np.array([0, 1])

    def fit(self, X, y):
        return self   # already trained

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.array(X, dtype=float) - self.x_min) / (self.x_max - self.x_min + 1e-8)
        t  = torch.tensor(Xs, dtype=torch.float32)
        with torch.no_grad():
            raw = self.torch_model(t).numpy()
        prob1 = torch.sigmoid(torch.tensor(raw)).numpy()
        return np.column_stack([1 - prob1, prob1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


sk_model     = SklearnWrapper(model, X_min, X_max)
X_train_df   = pd.DataFrame(X_train, columns=FEATURE_NAMES)

# %%
features_pdp = [0, 1, 2, 3, (0, 1), (0, 3)]
fig, ax = plt.subplots(figsize=(14, 10))
PartialDependenceDisplay.from_estimator(
    sk_model, X_train_df, features_pdp, feature_names=FEATURE_NAMES, ax=ax
)
plt.suptitle("Partial Dependence Plots", y=1.01)
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(14, 8))
PartialDependenceDisplay.from_estimator(
    sk_model, X_train_df, [0, 1, 2, 3], feature_names=FEATURE_NAMES, ax=ax,
    kind="individual"
)
plt.suptitle("ICE Plots", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Permutation feature importance
#
# Each criterion is shuffled independently across the test set and the accuracy drop is
# recorded.  A large drop indicates the model relies heavily on that criterion.

# %%
X_test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)

result = permutation_importance(
    sk_model, X_test_df, y_test,
    n_repeats=30, random_state=42, scoring='accuracy',
)

perm_df = pd.DataFrame({
    'criterion':       FEATURE_NAMES,
    'importance_mean': result.importances_mean,
    'importance_std':  result.importances_std,
}).sort_values('importance_mean', ascending=False)

print("Permutation feature importance (mean accuracy drop on test set):")
print(perm_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(
    perm_df['criterion'], perm_df['importance_mean'],
    xerr=perm_df['importance_std'], color='steelblue', capsize=4,
)
ax.set_xlabel("Mean accuracy decrease")
ax.set_title("Permutation Feature Importance")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Minimum single-criterion change to flip class (gradient-guided sampling)
#
# For each criterion we try two search directions (increasing and decreasing the value).
# We step uniformly from the current value toward the data boundary and stop at the
# first point where the predicted class changes.  The direction is tried in both ways
# to guarantee finding the global minimum over each single-criterion axis.

# %%
def find_min_flip_mlp(
    model:         nn.Module,
    x_scaled:      np.ndarray,
    x_orig:        np.ndarray,
    feature_names: list,
    feature_ranges: dict,
    x_min:         np.ndarray,
    x_max:         np.ndarray,
    n_steps:       int = 500,
):
    """Return the minimum-delta single-criterion flip for each criterion.

    Searches both ↑ and ↓ directions for each feature, keeps whichever direction
    reaches a class change with the smaller absolute delta.
    """
    with torch.no_grad():
        t = torch.tensor(x_scaled, dtype=torch.float32)
        current_class = int(model(t.unsqueeze(0)).squeeze().item() > 0)

    results = []

    for i, feat in enumerate(feature_names):
        lo_orig, hi_orig = feature_ranges[feat]
        lo_s = (lo_orig - x_min[i]) / (x_max[i] - x_min[i] + 1e-8)
        hi_s = (hi_orig - x_min[i]) / (x_max[i] - x_min[i] + 1e-8)
        orig_s = float(x_scaled[i])

        best = None
        for boundary in [hi_s, lo_s]:
            steps = np.linspace(orig_s, boundary, n_steps + 2)[1:]
            for new_s in steps:
                test_s = x_scaled.copy()
                test_s[i] = float(new_s)
                t2 = torch.tensor(test_s, dtype=torch.float32)
                with torch.no_grad():
                    new_out = model(t2.unsqueeze(0)).squeeze().item()
                new_class = int(new_out > 0)
                if new_class != current_class:
                    new_orig = float(new_s) * (x_max[i] - x_min[i] + 1e-8) + x_min[i]
                    delta = new_orig - float(x_orig[i])
                    if best is None or abs(delta) < abs(best['delta']):
                        best = {
                            'criterion': feat,
                            'delta':     delta,
                            'new_val':   new_orig,
                            'new_class': new_class,
                        }
                    break  # first flip found in this direction

        if best:
            results.append(best)

    return results, current_class


print("=" * 60)
for idx, lbl in zip(selected, labels):
    row  = df_pred.loc[idx]
    name = get_name(idx)
    pos  = df_pred.index.get_loc(idx)
    x_orig   = X_all[pos]
    x_scaled = X_all_s[pos]

    flips, curr_class = find_min_flip_mlp(
        model, x_scaled, x_orig, FEATURE_NAMES, FEATURE_RANGES, X_min, X_max
    )
    print(f"\n[{lbl}] {name}")
    print(f"  Current class: {curr_class}  "
          f"HP={row['HorsePower']:.0f} hp, Price=${row['Cars Prices']:.0f}, "
          f"Seats={row['Seats']:.0f}, Speed={row['Total Speed']:.0f} km/h")

    if flips:
        for f in flips:
            print(f"  → Change '{f['criterion']}' by {f['delta']:+.1f}"
                  f" to {f['new_val']:.1f}"
                  f" → predicted class flips to {f['new_class']}")
    else:
        print("  → No single-criterion change can flip the class within data bounds")

# %% [markdown]
# ## 14. Space sampling verification
#
# Apply the minimum change found and confirm the class flip.

# %%
print("=" * 60)
for idx, lbl in zip(selected, labels):
    row  = df_pred.loc[idx]
    name = get_name(idx)
    pos  = df_pred.index.get_loc(idx)
    x_orig     = X_all[pos]
    x_scaled   = X_all_s[pos]
    orig_pred  = int(row['pred'])

    flips, curr_class = find_min_flip_mlp(
        model, x_scaled, x_orig, FEATURE_NAMES, FEATURE_RANGES, X_min, X_max
    )

    print(f"\n[{lbl}] {name}  (original class={orig_pred})")
    if not flips:
        print("  No flip possible — nothing to verify.")
        continue

    best = min(flips, key=lambda x: abs(x['delta']))
    i_feat    = FEATURE_NAMES.index(best['criterion'])
    test_orig = x_orig.copy()
    test_orig[i_feat] = best['new_val']
    test_s = scale(test_orig)

    t_test = torch.tensor(test_s, dtype=torch.float32)
    with torch.no_grad():
        raw_out = model(t_test.unsqueeze(0)).squeeze().item()
    sampled_pred = int(raw_out > 0)
    sampled_prob = float(torch.sigmoid(torch.tensor(raw_out)).item())

    agree = "✓ AGREE" if sampled_pred != orig_pred else "✗ DISAGREE"
    print(f"  Analytical:  '{best['criterion']}' → {best['new_val']:.1f}"
          f"  (Δ={best['delta']:+.1f}) → class {best['new_class']}")
    print(f"  Sampling:    predicted class={sampled_pred}  prob={sampled_prob:.4f}  {agree}")

# %% [markdown]
# ## 15. Model interpretation summary
#
# * **Criterion influence**: Permutation importance and the SHAP beeswarm together reveal
#   which criteria drive predictions most.  A large accuracy drop when shuffling a criterion
#   confirms it is structurally important; large absolute SHAP values confirm it shifts
#   individual predictions significantly.
#
# * **Non-linearity and interactions**: Three ReLU hidden layers allow the MLP to capture
#   non-linear relationships and cross-criterion interactions that neither a linear model
#   nor a single decision tree can represent.  The 2-D PDP panels show where the effect
#   of one criterion depends on the level of another.
#
# * **No monotone constraint**: Unlike the XGBoost model, the MLP is unconstrained.
#   The PDP curves show empirically whether the learned function respects gain/cost
#   directions; any inversion (e.g. higher price → higher probability) signals that the
#   model overfit to noise in the constructed target labels.
#
# * **Thresholds / saturation**: Flat regions in PDP or ICE curves indicate ranges where
#   a further improvement in a criterion has no effect on the predicted preference score —
#   analogous to indifference intervals in classical MCDA.
#
# * **Criterion nature**: HorsePower, Seats and Total Speed are gain criteria (higher is
#   better).  Cars Prices is a cost criterion (lower is better).  The SHAP waterfall plots
#   for the three selected alternatives illustrate how each criterion pushes the score
#   above or below the decision threshold.
