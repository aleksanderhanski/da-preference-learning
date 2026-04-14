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
destination = "./dataset"  # ← change this to your preferred path
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
levels_desc = levels             # descending for labels (best → worst)

CRITERIA["Cars Prices"] = [f"≤{lvl}" for lvl in levels_desc[:-1]]

def snap_to_bin_label_cost(value, levels_asc, levels_desc):
    extended = np.append(levels_asc, np.inf)
    idx = np.digitize(value, extended) - 1
    idx = np.clip(idx, 0, len(levels_asc) - 2)
    # invert index so cheap = high bin
    inv_idx = len(levels_desc) - 2 - idx
    return f"≤{levels_desc[inv_idx]}"

df["Cars Prices"] = df["Cars Prices"].apply(
    lambda x: snap_to_bin_label_cost(x, levels_sorted, levels_desc)
)
print(df["Cars Prices"].unique())


# %% [markdown]
# # Dataset description:

# %% [markdown]
# 1. domain of problem is about choosing the best car for Michał.
#
# 2. source of data is kaggle dataset: https://www.kaggle.com/datasets/abdulmalik1518/cars-datasets-2025
#
# 3. Michał wants the car that is cheap (as he has not found internship yet), that is fast, but also important factor is that the car shall have many seats - Michał wants to do a walcome party in his car, and be able to invite as many people as possible.
#
# 4. Michał considered 31 alternatives - only cars with number of seats equal to 7 or more. But the Michał shrank his dataset to 8 random options from this so that he could use not only uta but also ahp on whole dataset and compare results. This is much less alternatives than in original dataset.
#
# 5. One alternative from original dataset is FERRARI SF90 STRADALE:
#
# - Engines: V8   
# - CC/Battery Capacity: 3990 cc 
# - HorsePower: 963 hp
# - Total Speed: 340 km/h 
# - Performance(0 - 100 )KM/H: 2.5 sec 
# - Cars Prices: $1,100,000 
# - Fuel Types: plug in hyrbrid
# - Seats: 2
# - Torque: 800 Nm
#
#   But Michał does not include this alternative, it has only 2 seats - it might be an acceptable choice for a date, but not for a legendary party that Michał wants to organize.
#
# 6. Michał considers four criteria: 
# - seats
# - car price
# - total speed
# - HorsePower
#
#   Michał also has some self-contradictory preferences regarding his favourite car models.
#
#   In the original data set there were more criteria: engines, CC/battery capacity, performance (0 - 100)km/h, fual types, torque, but Michał does not consider those criteria. Four criteria that Michał chose are fully sufficient.
#
# 7. 
# - seats is discrete gain type criterium
# - total speed is gain type, it was continuous, we divided it into 5 categories: ['≥155.0', '≥168.0', '≥181.0', '≥194.0', '≥207.0']
#   car price and HorsePower also were continuous, we divided them into categories similar way. 
#
#   For price, which was cost type, we inverted ordering, so that the cheapest correspond to best.
#   
#   While transforming continuous categories into discrete ones, we did not take outliers, as we saw that in each case there were not that many (Ferrari would be outlier compared to cars Michał chose from, but we did not consider it). In each case we did divide distirbution linearly into categories.
#
# 8. All 4 criteria that were considered are of equal importance. Other criteria are irrelevant.
#
# 10. In my opinion the best alternative would be Ferrari, but Michał wants to consider alternative that has 7 seats or more, if that constraint is fulfilled, then he wants to consider number of seats, horse power, total speed, and price as similarly important.
#
# 11. There is no alternative that seems to be much better than the others. One alternative that can be considered is Honda Pilot - it costs 40,000$, so is realtively cheap, total speed of 209 km/h, what is better score than average among considered alternatives, and its horse power is not worse than many other alternatives. So it is good not because of one outsanding criterium, but because of good overall value of criteria.
#
# 12. One of alternatives that seems not that strong is Kia Carnival EX. It is quite cheap, and decently fast, but has quite small horse power. It is not a very bad car, and there is no one major disadvantage.
#
# 13. 
#
# - Honda PILOT vs Nissan NV1500: PILOT is much faster (209 vs 160 km/h) and has more HP (285 vs 261), but NV1500 can fit 12 people compared to just 8.
#
# - Nissan NV1500 vs VW Transporter: NV1500 has more seats (12 vs 9) and more HP (261 vs 153), but Transporter is a bit faster (180 vs 160 km/h).
#
# - Ferrari vs Nissan NV1500: Ferrari can fit only 2 people compared to 12.


# %%
def save_preprocessed(dataframe, path="dataset/dataset_preprocessed.csv"):
    dataframe.to_csv(path, index=False)
    print(f"Preprocessed dataset saved to: {path}")

save_preprocessed(df)

# %%
