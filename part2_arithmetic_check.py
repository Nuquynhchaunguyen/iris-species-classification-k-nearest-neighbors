"""Optional arithmetic-only check of the six invented Part 2 flowers.

Run ``python part2_arithmetic_check.py``. This does not load real Iris data or
use StandardScaler, KNeighborsClassifier, or any nearest-neighbor search API.
The hand calculations in memo.md are the submitted explanation.
"""

from collections import Counter

import numpy as np


IDS = ("A", "B", "C", "D", "E", "F")
SPECIES = ("setosa", "setosa", "versicolor", "versicolor", "virginica", "virginica")
TRAINING = np.array([[2, 1], [2, 2], [4, 1], [4, 2], [6, 1], [6, 2]], dtype=float)
QUERY = np.array([4.7, 1.8], dtype=float)


def show_order(title: str, distances: np.ndarray) -> list[int]:
    order = sorted(range(len(IDS)), key=lambda index: (distances[index], index))
    print(f"\n{title}")
    for rank, index in enumerate(order, start=1):
        print(f"  {rank}. {IDS[index]} {SPECIES[index]:10s} distance={distances[index]:.4f}")
    for k in (1, 3):
        voters = order[:k]
        counts = Counter(SPECIES[index] for index in voters)
        winner = max(counts, key=counts.get)
        print(f"  k={k}: voters={[IDS[index] for index in voters]}, votes={dict(counts)}, winner={winner}")
    return order


def main() -> None:
    n = len(TRAINING)
    sums = TRAINING.sum(axis=0)
    means = sums / n
    deviations = TRAINING - means
    squared_deviations = deviations**2
    sums_of_squares = squared_deviations.sum(axis=0)
    population_variances = sums_of_squares / n
    population_sds = np.sqrt(population_variances)

    print("Invented Part 2 training statistics; Q is excluded")
    for feature, column in (("L", 0), ("W", 1)):
        print(
            f"  {feature}: sum={sums[column]:.4f}, mean={means[column]:.4f}, "
            f"sum of squares={sums_of_squares[column]:.4f}, "
            f"population variance={population_variances[column]:.4f}, "
            f"SD={population_sds[column]:.4f}"
        )
    print("\nID | L deviation | L square | W deviation | W square")
    for index, name in enumerate(IDS):
        print(
            f" {name} | {deviations[index, 0]:>11.4f} | "
            f"{squared_deviations[index, 0]:>8.4f} | "
            f"{deviations[index, 1]:>11.4f} | "
            f"{squared_deviations[index, 1]:>8.4f}"
        )

    # Apply the SAME unrounded training statistics to all references and Q.
    all_points = np.vstack((TRAINING, QUERY))
    scaled = (all_points - means) / population_sds
    print("\nScaled coordinates (training statistics only)")
    for name, row in zip((*IDS, "Q"), scaled):
        print(f"  {name}: zL={row[0]: .4f}, zW={row[1]: .4f}")

    scaled_distances = np.sqrt(((scaled[:n] - scaled[n]) ** 2).sum(axis=1))
    raw_distances = np.sqrt(((TRAINING - QUERY) ** 2).sum(axis=1))
    scaled_order = show_order("Scaled Euclidean distances", scaled_distances)
    raw_order = show_order("Raw-centimeter Euclidean distances", raw_distances)

    np.testing.assert_allclose(means, [4.0, 1.5])
    np.testing.assert_allclose(population_variances, [8 / 3, 0.25])
    assert [IDS[index] for index in scaled_order] == ["D", "F", "C", "B", "E", "A"]
    assert [IDS[index] for index in raw_order] == ["D", "C", "F", "E", "B", "A"]
    print("\nArithmetic checks passed.")


if __name__ == "__main__":
    main()
