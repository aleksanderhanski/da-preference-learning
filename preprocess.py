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
# **Aleksander Hański 160315 and Michał Żurawski 160252**

# %%
import kagglehub
import shutil
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from itertools import combinations
import pulp
from pulp import (
    LpProblem, LpMinimize, LpMaximize, LpVariable, LpStatus,
    lpSum, value, GLPK_CMD
)
import sys

# %% [markdown]
# # Loading dataset:

# %%
# Download to kaggle cache first
cache_path = kagglehub.dataset_download("abdulmalik1518/cars-datasets-2025")

# Copy to your desired local folder
destination = "./dataset"
os.makedirs(destination, exist_ok=True)
shutil.copytree(cache_path, destination, dirs_exist_ok=True)

print("Files saved to:", destination)
for f in os.listdir(destination):
    print(" -", f)

# %%
df = pd.read_csv("./dataset/Cars Datasets 2025.csv", encoding="cp1252")

df.head()

# %% [markdown]
# # Dataset preprocessing:

# %%
print(df.isnull().sum())

# %%
df = df.dropna()

# %%
for col in df.columns:
    print(f"\n{col} ({df[col].nunique()} unique values) | type: {df[col].dtype}")
    if df[col].nunique() < 20: 
        print(df[col].unique())

# %% [markdown]
# ## Seats:

# %%
print(df[df["Seats"].isin(["2+2", "2-6", "2-7","2-9", "2-12", "2-15", "7-8"])])

# %% [markdown]
# There are only alternatives with "2+2" seats.

# %%
df = df[df["Seats"].isin(['7','8','9','12'])]

print(f'Number of alternatives after filtering by number of seats: {len(df)}')

# %%
CRITERIA = {"Seats": ['7','8','9','12']}

# %% [markdown]
# ## Total speed:

# %%
print(df["Total Speed"])

# %%
df["Total Speed"] = df["Total Speed"].str.replace(" km/h", "").astype(int)
print(df["Total Speed"])

# %%
print(df["Total Speed"].min())
print(df["Total Speed"].max())

df["Total Speed"].hist(bins=20)
plt.title("Total Speed Distribution")
plt.xlabel("Speed (km/h)")
plt.ylabel("Count")
plt.show()

# %%
levels = np.linspace(df["Total Speed"].min(), df["Total Speed"].max(), 6)
levels

# %%
CRITERIA["Total Speed"] = [f"≥{lvl}" for lvl in levels[:-1]]
print(CRITERIA)


# %%
def snap_to_bin_label(value, levels):
    extended = np.append(levels, np.inf)
    idx = np.digitize(value, extended) - 1
    idx = np.clip(idx, 0, len(levels) - 1)
    # use levels[idx] but cap at second-to-last level
    label_idx = min(idx, len(levels) - 2)
    return f"≥{levels[label_idx]}"

df["Total Speed_cont"] = df["Total Speed"]  # save continuous before discretization
df["Total Speed"] = df["Total Speed"].apply(lambda x: snap_to_bin_label(x, levels))
print(df["Total Speed"].unique())

# %% [markdown]
# ## HorsePower:

# %%
print(df["HorsePower"])

# %%
df["HorsePower"] = (df["HorsePower"]
    .str.replace(" hp", "", case=False)
    .str.replace(",", "")
    .str.strip()
    .str.split(" - ")
    .apply(lambda x: round((int(x[0]) + int(x[1])) / 2) if len(x) == 2 else int(x[0]))
)
print(df["HorsePower"])

# %%
print(df["HorsePower"].min())
print(df["HorsePower"].max())

df["HorsePower"].hist(bins=20)
plt.title("HorsePower Distribution")
plt.xlabel("HorsePower)")
plt.ylabel("Count")
plt.show()

df = df[df["HorsePower"] < 682]  # remove Cadillac Escalade V outlier (682 hp)

# %%
levels = np.round(np.linspace(df["HorsePower"].min(), df["HorsePower"].max(), 6), 2)

CRITERIA["HorsePower"] = [f"≥{lvl}" for lvl in levels[:-1]]

df["HorsePower_cont"] = df["HorsePower"]  # save continuous before discretization
df["HorsePower"] = df["HorsePower"].apply(lambda x: snap_to_bin_label(x, levels))

print(CRITERIA)
print(df["HorsePower"].unique())

# %% [markdown]
# ## Price:

# %%
print(df["Cars Prices"])

# %%
df["Cars Prices"] = (df["Cars Prices"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "")
    .str.strip()
    .str.split(" - ")
    .apply(lambda x: (int(x[0]) + int(x[1])) / 2 if len(x) == 2 else int(x[0]))
)
print(df["Cars Prices"])

# %%
print(df["Cars Prices"].min())
print(df["Cars Prices"].max())

df["Cars Prices"].hist(bins=20)
plt.title("Cars Prices Distribution")
plt.xlabel("Cars Prices")
plt.ylabel("Count")
plt.show()

# %%
levels = np.linspace(df["Cars Prices"].max(), df["Cars Prices"].min(), 6)

levels_sorted = np.sort(levels)  # ascending for digitize
levels_desc = levels             # descending for labels (best -> worst)

CRITERIA["Cars Prices"] = [f"≤{lvl}" for lvl in levels_desc[:-1]]

def snap_to_bin_label_cost(value, levels_asc, levels_desc):
    extended = np.append(levels_asc, np.inf)
    idx = np.digitize(value, extended) - 1
    idx = np.clip(idx, 0, len(levels_asc) - 2)
    # invert index so cheap = high bin
    inv_idx = len(levels_desc) - 2 - idx
    return f"≤{levels_desc[inv_idx]}"

df["Cars Prices_cont"] = df["Cars Prices"]  # save continuous before discretization
df["Cars Prices"] = df["Cars Prices"].apply(
    lambda x: snap_to_bin_label_cost(x, levels_sorted, levels_desc)
)
print(df["Cars Prices"].unique())


# %% [markdown]
# # Dataset description:

# %% [markdown]
# 1. The domain of the problem is choosing the best car for Michał.
#
# 2. Source of data: Kaggle dataset https://www.kaggle.com/datasets/abdulmalik1518/cars-datasets-2025
#
# 3. Michał wants a car that is cheap (as he has not found an internship yet), fast, but also with many seats — Michał wants to organise a welcome party in his car and invite as many people as possible.
#
# 4. After preprocessing, the dataset contains **177 alternatives**. Only cars with a number of seats in {7, 8, 9, 12} were kept (alternatives with non-standard seat counts such as "2+2", "2-6", etc. were excluded). One additional outlier was removed: the Cadillac Escalade V (682 hp), whose HorsePower value was far above all other alternatives and would distort the learned preference model.
#
# 5. One example of an excluded alternative is the FERRARI SF90 STRADALE:
#
# - Engines: V8
# - CC/Battery Capacity: 3990 cc
# - HorsePower: 963 hp
# - Total Speed: 340 km/h
# - Performance (0–100 km/h): 2.5 sec
# - Cars Prices: $1,100,000
# - Fuel Types: plug-in hybrid
# - Seats: 2
# - Torque: 800 Nm
#
#   The Ferrari is excluded because it has only 2 seats — it might be an acceptable choice for a date, but not for the legendary party Michał wants to organise.
#
# 6. Michał considers four criteria:
# - Seats
# - Cars Prices
# - Total Speed
# - HorsePower
#
#   The original dataset contained additional attributes (engine type, CC/battery capacity, 0–100 km/h performance, fuel type, torque), but Michał does not consider those. The four chosen criteria are fully sufficient to capture his preferences.
#
# 7. Criterion types and scales:
# - **Seats** — discrete **gain** criterion; values in {7, 8, 9, 12}.
# - **Total Speed** — continuous **gain** criterion (km/h); higher speed is preferred.
# - **HorsePower** — continuous **gain** criterion (hp); higher power is preferred.
# - **Cars Prices** — continuous **cost** criterion (USD); lower price is preferred.
#
#   All three continuous criteria are used at their raw continuous values in the preference-learning models (saved in `dataset_preprocessed_continuous.csv`).
#
# 8. All four criteria are treated as equally important. Other attributes available in the raw dataset are irrelevant to Michał's decision.
#
# 9. There is no single alternative that dominates all others. One strong candidate from the dataset is the **Honda Pilot** — it is relatively cheap (~$40,000), has a total speed of 209 km/h (above average among the considered alternatives), and its HorsePower is comparable to many competitors. It is not outstanding on any single criterion, but offers good overall value across all four.
#
# 10. An example of a weaker alternative is the **Kia Carnival EX**. It is affordable and decently fast, but its HorsePower is below average. It has no single decisive disadvantage, but no decisive advantage either.
#
# 11. A few illustrative pairwise comparisons:
#
# - **Honda Pilot vs Nissan NV1500**: Pilot is faster (209 vs 160 km/h) and has more HP (285 vs 261), but the NV1500 seats 12 people compared to just 8.
#
# - **Nissan NV1500 vs VW Transporter**: NV1500 seats more people (12 vs 9) and has more HP (261 vs 153), but the Transporter is slightly faster (180 vs 160 km/h).
#
# - **Ferrari SF90 vs Nissan NV1500**: The Ferrari is dramatically faster and more powerful, but seats only 2 people vs 12 — making it unsuitable for Michał's use case.


# %%
def save_preprocessed(dataframe,
                       path_continuous="dataset/dataset_preprocessed_continuous.csv"):
    cols_continuous = ["Company Names", "Cars Names", "HorsePower_cont", "Cars Prices_cont", "Seats", "Total Speed_cont"]
    (dataframe[cols_continuous]
        .rename(columns={"HorsePower_cont": "HorsePower",
                         "Cars Prices_cont": "Cars Prices",
                         "Total Speed_cont": "Total Speed"})
        .to_csv(path_continuous, index=False))
    print(f"Continuous dataset saved to:  {path_continuous}")

save_preprocessed(df)

# %%
