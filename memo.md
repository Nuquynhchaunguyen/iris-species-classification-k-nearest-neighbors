# Iris Species Classification: From Measurements to Neighbor Votes

Part 2 uses invented records; Parts 3–5 use real Iris data. Part 6A's hypothetical counts are separate from the measured experiment.

## Part 1 — What KNN does

An **input feature** is a measurement (petal length or width); a **target label** is a known species; a **query** is the flower to classify. **k** is the number of voting references, not the number of species. `StandardScaler.fit` learns training feature means and population standard deviations without labels. `KNeighborsClassifier.fit` stores scaled references and their labels; prediction uses labels.

KNN sorts eligible references by query distance and takes the nearest k; farther labels do not vote. Ordinary KNN has no iterative loss-minimizing coefficient updates. Fitting ends after storing references, tuning after comparing predeclared validation candidates, and prediction after neighbor voting. A three-species classifier cannot discover a fourth species; clustering might suggest new groups, but experts must rule out noise and ordinary variation.

## Part 2 — Complete hand calculation on invented records

The six training lengths sum to 24 cm and widths to 9 cm. Q = (4.7, 1.8) is excluded from training statistics.

| ID | L | L − 4 | (L − 4)² | W | W − 1.5 | (W − 1.5)² | Known species |
|:--|--:|--:|--:|--:|--:|--:|:--|
| A | 2.0 | −2.0 | 4.0 | 1.0 | −0.5 | 0.25 | setosa |
| B | 2.0 | −2.0 | 4.0 | 2.0 | +0.5 | 0.25 | setosa |
| C | 4.0 | 0.0 | 0.0 | 1.0 | −0.5 | 0.25 | versicolor |
| D | 4.0 | 0.0 | 0.0 | 2.0 | +0.5 | 0.25 | versicolor |
| E | 6.0 | +2.0 | 4.0 | 1.0 | −0.5 | 0.25 | virginica |
| F | 6.0 | +2.0 | 4.0 | 2.0 | +0.5 | 0.25 | virginica |

| Feature | Sum | Mean | Sum of squared deviations | Population variance (divide by 6) | Population SD |
|:--|--:|--:|--:|--:|--:|
| L | 24.0000 | 24/6 = 4.0000 | 16.0000 | 16/6 = 8/3 = 2.6667 | √(8/3) = 1.6330 |
| W | 9.0000 | 9/6 = 1.5000 | 1.5000 | 1.5/6 = 0.2500 | √0.25 = 0.5000 |

All subsequent arithmetic uses unrounded values 4, 1.5, √(8/3), and 0.5; displayed answers use four decimals. For A and Q:

```text
A: zL = (2 − 4)/√(8/3) = −1.2247; zW = (1 − 1.5)/0.5 = −1.0000
Q: zL = (4.7 − 4)/√(8/3) = 0.4287; zW = (1.8 − 1.5)/0.5 = 0.6000
```

| Record | zL | zW |
|:--|--:|--:|
| A | −1.2247 | −1.0000 |
| B | −1.2247 | +1.0000 |
| C | 0.0000 | −1.0000 |
| D | 0.0000 | +1.0000 |
| E | +1.2247 | −1.0000 |
| F | +1.2247 | +1.0000 |
| Q | +0.4287 | +0.6000 |

For D, every scaled-distance step is:

```text
zD − zQ = (0 − 0.428660704987, 1 − 0.6) = (−0.428660704987, 0.4)
Squared terms = (−0.428660704987)² = 0.18375; (0.4)² = 0.16
Sum = 0.18375 + 0.16 = 0.34375
Distance = √0.34375 = 0.5863
```

All scaled distances, nearest first:

| Rank | Reference | Species | Squared differences and square root | Scaled distance to Q |
|--:|:--|:--|:--|--:|
| 1 | D | versicolor | √(0.18375 + 0.16000) | 0.5863 |
| 2 | F | virginica | √(0.63375 + 0.16000) | 0.8909 |
| 3 | C | versicolor | √(0.18375 + 2.56000) | 1.6564 |
| 4 | B | setosa | √(2.73375 + 0.16000) | 1.7011 |
| 5 | E | virginica | √(0.63375 + 2.56000) | 1.7871 |
| 6 | A | setosa | √(2.73375 + 2.56000) | 2.3008 |

| k | Voters | Votes (setosa, versicolor, virginica) | Prediction |
|--:|:--|:--|:--|
| 1 | D | (0, 1, 0) | versicolor |
| 3 | D, F, C | (0, 2, 1) | versicolor |

At k = 3, the winning fraction is 2/3 = **66.6667%** of neighbor votes. This is not a calibrated correctness probability and is not fed into a coefficient update.

Using raw centimeters instead gives these six calculations and a different order:

| Rank | Reference | Species | Raw calculation | Raw distance (cm) |
|--:|:--|:--|:--|--:|
| 1 | D | versicolor | √[(4 − 4.7)² + (2 − 1.8)²] = √0.53 | 0.7280 |
| 2 | C | versicolor | √[(4 − 4.7)² + (1 − 1.8)²] = √1.13 | 1.0630 |
| 3 | F | virginica | √[(6 − 4.7)² + (2 − 1.8)²] = √1.73 | 1.3153 |
| 4 | E | virginica | √[(6 − 4.7)² + (1 − 1.8)²] = √2.33 | 1.5264 |
| 5 | B | setosa | √[(2 − 4.7)² + (2 − 1.8)²] = √7.33 | 2.7074 |
| 6 | A | setosa | √[(2 − 4.7)² + (1 − 1.8)²] = √7.93 | 2.8160 |

Raw k = 3 uses D, C, F and still predicts versicolor 2–1. C/F and B/E switch order because standardization weights a width difference more: width SD is 0.5 versus length SD 1.6330, despite shared centimeter units. Training means and SDs prepare distances but cannot select or tally neighbors. Without Q's true label, its accuracy is unknown. Odd k can still tie across three classes: k = 3 permits 1–1–1 votes, so a tie rule is needed.

## Parts 3–4 — Real Iris data, partitioning, and k selection

Scikit-learn's `load_iris().DESCR` attributes its 150-row Iris dataset to Fisher. It contains 50 each of setosa (0), versicolor (1), and virginica (2). The only inputs, in order, are `petal length (cm)` and `petal width (cm)`; each has zero missing values. Species codes and row IDs are never distance features or averaged predictions. Stratified index splits with `random_state=42` reserve 20% for test, then 25% of development for validation, retaining returned order.

| Partition | Rows | setosa | versicolor | virginica | Purpose |
|:--|--:|--:|--:|--:|:--|
| Training | 90 | 30 | 30 | 30 | Fit candidate scalers and KNN references |
| Validation | 30 | 10 | 10 | 10 | Choose k without fitting on these rows |
| Test | 30 | 10 | 10 | 10 | One final estimate after choices are frozen |

The index sets are pairwise disjoint and cover 150 rows. Test measurements do not fit development scalers; test labels do not choose k. Every candidate uses training-only `StandardScaler()` and uniform, brute-force Euclidean KNN (`metric="minkowski", p=2`) on the fixed split. Scores appear in [the validation CSV](reports/knn_validation.csv) and [plot](reports/k_selection.png).

| k | Training accuracy | Validation accuracy |
|--:|--:|--:|
| 1 | 1.000000 | 0.900000 |
| 3 | 0.988889 | 0.933333 |
| 5 | 0.988889 | 0.933333 |
| 7 | 0.977778 | 0.933333 |
| 9 | 0.977778 | 0.933333 |
| 15 | 0.977778 | 0.933333 |
| 25 | 0.966667 | 0.933333 |

All k from 3 through 25 achieve 28/30 validation predictions, so exact counts and the smallest-k tie rule select **k = 3** before test evaluation. Training accuracy instead favors k = 1: each training row can be its own zero-distance neighbor. Its 100% training versus 90% validation score suggests overfitting. At k = 3, the gap is 5.56 points with higher validation accuracy; larger k shows no clear underfitting pattern. A small gap alone cannot prove good generalization. Seven separately fitted pipelines change k, not the training references or other settings; these are not gradient updates. Search ends after the fixed comparison.

For tied class votes, this scikit-learn version chooses the lowest encoded class. A distance tie at the kth boundary is different: row order and library version can change which reference enters the vote.

## Part 5 — Frozen refit and held-out test

With k frozen at 3, a **new** scaler/KNN pipeline fits all 120 development rows in original order. A most-frequent `DummyClassifier` fits those rows without scaling. Both are tested once; these are refitted-model scores, distinct from Part 4.

| Model | Test accuracy | Test macro-F1 |
|:--|--:|--:|
| KNN | 0.966667 | 0.966583 |
| Most-frequent baseline | 0.333333 | 0.166667 |

Confusion matrices use **actual rows and predicted columns** in setosa, versicolor, virginica order; see [the labeled figure](reports/test_confusion_matrices.png).

| Actual \ predicted, KNN | setosa | versicolor | virginica |
|:--|--:|--:|--:|
| setosa | 10 | 0 | 0 |
| versicolor | 0 | 9 | 1 |
| virginica | 0 | 0 | 10 |

| Actual \ predicted, baseline | setosa | versicolor | virginica |
|:--|--:|--:|--:|
| setosa | 10 | 0 | 0 |
| versicolor | 10 | 0 | 0 |
| virginica | 10 | 0 | 0 |

| Model | Species | Precision | Recall | F1 | Support |
|:--|:--|--:|--:|--:|--:|
| KNN | setosa | 1.000000 | 1.000000 | 1.000000 | 10 |
| KNN | versicolor | 1.000000 | 0.900000 | 0.947368 | 10 |
| KNN | virginica | 0.909091 | 1.000000 | 0.952381 | 10 |
| Baseline | setosa | 0.333333 | 1.000000 | 0.500000 | 10 |
| Baseline | versicolor | 0.000000 | 0.000000 | 0.000000 | 10 |
| Baseline | virginica | 0.000000 | 0.000000 | 0.000000 | 10 |

See [the test summary](reports/test_summary.csv) and [per-species CSV](reports/test_per_species.csv). Accuracy counts the fraction of all correct predictions; macro-F1 averages species F1 equally. KNN mistakes one actual versicolor for virginica. The baseline calls all 30 setosa, missing 20 flowers; KNN exceeds it on both summary metrics. With `zero_division=0`, a zero baseline precision for an unpredicted species represents an undefined denominator. Adding validation references can alter neighborhoods, while refitting the scaler changes distances even at fixed k. These scores establish neither field performance nor causation nor detection of unknown species.

## Part 6A — Hypothetical matrix, calculated separately

| Actual \ predicted | setosa | versicolor | virginica |
|:--|--:|--:|--:|
| setosa | 18 | 2 | 0 |
| versicolor | 1 | 15 | 4 |
| virginica | 0 | 3 | 17 |

```text
Total = 60; correct = 18 + 15 + 17 = 50; errors = 60 − 50 = 10
Accuracy = 50/60 = 0.8333
Virginica vs rest: TP = 17, FP = 4, FN = 3, TN = 36
Precision = TP/(TP + FP) = 17/(17 + 4) = 17/21 = 0.8095
Recall = TP/(TP + FN) = 17/(17 + 3) = 17/20 = 0.8500
F1 = 2PR/(P + R) = 2(17/21)(17/20)/[(17/21) + (17/20)] = 34/41 = 0.8293
```

Of the 21 flowers *predicted* virginica, 17 actually are; of the 20 *actual* virginica flowers, 17 are found. A false positive might prompt unnecessary follow-up for versicolor; a false negative might overlook virginica. Virginica F1 combines one species' precision and recall, unlike overall accuracy or macro-F1 across all species.

## Part 6B — Reasoning about 85 million rows

Validation queries can search **51 million** training references; after refit, test queries can search **68 million** development references. Including all 85 million during validation would leak validation/test rows. Keeping the global nearest 25 after each exhaustive validation query allows scoring every prescribed k from nested neighbor prefixes, without seven full scans.

```text
Validation comparisons = 17,000,000 × 51,000,000 = 867,000,000,000,000
Validation query batches = 17,000,000 / 10,000 = 1,700
```

Each query in 1,700 batches of 10,000 still searches every eligible reference; batching limits memory, not total exhaustive comparisons. Chunked scaling instead combines training-only counts, sums, and sums of squares into **global** feature statistics. Gradient-descent minibatches update coefficients; KNN has no such updates or `partial_fit`. Per-chunk scalers create inconsistent coordinates, while successive KNN `fit` calls replace rather than accumulate references.

For reference blocks, merge the **global** nearest 25 before voting; the last block alone may omit nearer flowers. KNN has no universal row limit: dimensions, memory, latency, and search strategy matter. An exact index may inspect fewer distances while every eligible reference remains a candidate. Approximate search may miss true neighbors; compare neighbor recall and classification metrics against exact search on a representative sample, alongside time and memory.

## Research process and source acknowledgment

One design question was whether two features measured in centimeters must preserve neighbor order after scaling. I checked every raw and standardized distance using the unrounded population SD and independently recalculated the Part 2 statistics with Python/NumPy arithmetic in [the arithmetic check](part2_arithmetic_check.py); the changed C/F and B/E positions show why the shared unit does not settle the issue. I also compared integer validation correct counts before applying the smallest-k tie rule and checked the final confusion matrices against the per-species metrics. This report and the analysis code were prepared with assistance from OpenAI Codex; the calculations and claims should be reviewed and understood by the submitter before submission.

Sources: [KNN homework instructions](https://github.com/shylasolis/machine-learning-pathway/blob/main/Project_02_Classification/k-nearest-neighbors/homework_instructions.md), [student rubric](https://github.com/shylasolis/machine-learning-pathway/blob/main/Project_02_Classification/k-nearest-neighbors/homework_rubric.md), the course KNN beginner guide and comparison notebook, and the scikit-learn `load_iris().DESCR` dataset description and [KNeighborsClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). The measured experiment was checked with Python 3.14.10, NumPy 2.5.4, Matplotlib 3.11.2, and scikit-learn 1.9.1. Independent classmate replication and the private validation form still require human completion after the final repository commit.
