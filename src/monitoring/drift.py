"""Расчет Population Stability"""

import numpy as np


def psi_calculation(ref, new, bins=10):
    """Расчет Population Stability"""
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)):
        raise ValueError("bins должен быть целым числом не меньше 2")
    if bins < 2:
        raise ValueError("bins должен быть целым числом не меньше 2")

    ref = np.asarray(ref, dtype=float)
    new = np.asarray(new, dtype=float)

    if ref.ndim != 1 or new.ndim != 1:
        raise ValueError("Массивы должны быть одномерными")

    if ref.size == 0 or new.size == 0:
        raise ValueError("Массивы не должны быть пустыми")

    if not np.isfinite(ref).all() or not np.isfinite(new).all():
        raise ValueError("Данные содержат NaN или бесконечность")

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(ref, quantiles))

    if len(breakpoints) == 1:
        value = breakpoints[0]
        ref_counts = np.array([0, ref.size, 0])
        new_counts = np.array(
            [(new < value).sum(), (new == value).sum(), (new > value).sum()]
        )
    else:
        if len(breakpoints) == 2:
            # Keep two distinct reference values in separate bins.
            midpoint = breakpoints[0] / 2 + breakpoints[1] / 2
            breakpoints = np.array([-np.inf, midpoint, np.inf])
        else:
            # Include observations outside the reference range.
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf
        ref_counts = np.histogram(ref, bins=breakpoints)[0]
        new_counts = np.histogram(new, bins=breakpoints)[0]

    # Bins empty in both samples carry no information and need no smoothing.
    occupied = (ref_counts > 0) | (new_counts > 0)
    ref_pct = ref_counts[occupied] / len(ref)
    new_pct = new_counts[occupied] / len(new)

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    new_pct = np.clip(new_pct, epsilon, None)
    ref_pct /= ref_pct.sum()
    new_pct /= new_pct.sum()

    psi = np.sum((ref_pct - new_pct) * np.log(ref_pct / new_pct))

    return float(psi)


if __name__ == "__main__":
    np.random.seed(42)

    train = np.random.normal(0, 1, 10000)

    # Распределение почти не изменилось
    new_normal = np.random.normal(0, 1, 1000)

    # Распределение изменилось
    new_drifted = np.random.normal(1, 1, 1000)

    print("PSI без дрифта:", psi_calculation(train, new_normal))
    print("PSI с дрифтом:", psi_calculation(train, new_drifted))
