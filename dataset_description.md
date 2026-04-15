1. The domain of the problem is choosing the best car for Michał.

2. Source of data: Kaggle dataset https://www.kaggle.com/datasets/abdulmalik1518/cars-datasets-2025

3. Michał wants a car that is cheap (as he has not found an internship yet), fast, but also with many seats — Michał wants to organise a welcome party in his car and invite as many people as possible.

4. After preprocessing, the dataset contains **177 alternatives**. Only cars with a number of seats in {7, 8, 9, 12} were kept (alternatives with non-standard seat counts such as "2+2", "2-6", etc. were excluded). One additional outlier was removed: the Cadillac Escalade V (682 hp), whose HorsePower value was far above all other alternatives and would distort the learned preference model.

5. One example of an excluded alternative is the FERRARI SF90 STRADALE:

- Engines: V8
- CC/Battery Capacity: 3990 cc
- HorsePower: 963 hp
- Total Speed: 340 km/h
- Performance (0–100 km/h): 2.5 sec
- Cars Prices: $1,100,000
- Fuel Types: plug-in hybrid
- Seats: 2
- Torque: 800 Nm

  The Ferrari is excluded because it has only 2 seats — it might be an acceptable choice for a date, but not for the legendary party Michał wants to organise.

6. Michał considers four criteria:
- Seats
- Cars Prices
- Total Speed
- HorsePower

  The original dataset contained additional attributes (engine type, CC/battery capacity, 0–100 km/h performance, fuel type, torque), but Michał does not consider those. The four chosen criteria are fully sufficient to capture his preferences.

7. Criterion types and scales:
- **Seats** — discrete **gain** criterion; values in {7, 8, 9, 12}.
- **Total Speed** — continuous **gain** criterion (km/h); higher speed is preferred.
- **HorsePower** — continuous **gain** criterion (hp); higher power is preferred.
- **Cars Prices** — continuous **cost** criterion (USD); lower price is preferred.

  All three continuous criteria are used at their raw continuous values in the preference-learning models (saved in `dataset_preprocessed_continuous.csv`).

8. All four criteria are treated as equally important. Other attributes available in the raw dataset are irrelevant to Michał's decision.

9. There is no single alternative that dominates all others. One strong candidate from the dataset is the **Honda Pilot** — it is relatively cheap (~$40,000), has a total speed of 209 km/h (above average among the considered alternatives), and its HorsePower is comparable to many competitors. It is not outstanding on any single criterion, but offers good overall value across all four.

10. An example of a weaker alternative is the **Kia Carnival EX**. It is affordable and decently fast, but its HorsePower is below average. It has no single decisive disadvantage, but no decisive advantage either.

11. A few illustrative pairwise comparisons:

- **Honda Pilot vs Nissan NV1500**: Pilot is faster (209 vs 160 km/h) and has more HP (285 vs 261), but the NV1500 seats 12 people compared to just 8.

- **Nissan NV1500 vs VW Transporter**: NV1500 seats more people (12 vs 9) and has more HP (261 vs 153), but the Transporter is slightly faster (180 vs 160 km/h).

- **Ferrari SF90 vs Nissan NV1500**: The Ferrari is dramatically faster and more powerful, but seats only 2 people vs 12 — making it unsuitable for Michał's use case.
