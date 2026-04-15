# Project Preference Learning - Cars 2025

Aleksander Hański 160315 and Michał Żurawski 160252.

Comparative analysis of preference-learning models on a car-selection
dataset (4 monotone criteria, 177 alternatives, binary class).

## Models

| File                | Model                             | Grade tier |
| ------------------- | --------------------------------- | ---------- |
| `xgboost.ipynb`     | XGBoost with monotone constr.     | 3          |
| `ann_utadis.ipynb`  | ANN-UTADIS (additive utility)     | 4          |
| `deep_nn.ipynb`     | Deep feed-forward NN (BCE + Adam) | 5          |

Each notebook reports Accuracy / F1 / AUC (4 dp) and contains:
feature importance (gain / Permutation FI), PDP, ICE, SHAP (waterfalls +
beeswarm), selection of three alternatives (preferred / borderline /
not-preferred), analytical _and_ sampling minimum-change class flip, and an
interpretation summary.

All three notebooks share helpers from `common_cars.py` (data loading,
metrics, PFI, PDP/ICE, model-agnostic min-flip).

## Dataset

Preprocessed by `preprocess.py` - downloads the Kaggle
`abdulmalik1518/cars-datasets-2025` dataset, filters to cars with ≥7 seats,
cleans unit strings, removes one HP outlier, and writes:

- `dataset/dataset_preprocessed_continuous.csv` - raw numerical values used by all three models

## Run

```bash
# one-time: download + preprocess
python preprocess.py

# run notebooks interactively
jupyter notebook xgboost.ipynb
jupyter notebook ann_utadis.ipynb
jupyter notebook deep_nn.ipynb
```

## Dependencies

`numpy pandas matplotlib seaborn scikit-learn xgboost shap torch kagglehub`
