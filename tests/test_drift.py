"""Check PSI distributions, degenerate references, and invalid inputs."""

import numpy as np
import pytest

from src.monitoring.drift import psi_calculation


def test_identical_distributions_have_zero_psi():
    ref = np.arange(100, dtype=float)
    assert psi_calculation(ref, np.tile(ref, 2)) == pytest.approx(0)


def test_shift_and_values_outside_reference_range_are_detected():
    ref = np.arange(100, dtype=float)
    assert psi_calculation(ref, ref + 1000) > 0
    assert psi_calculation(ref, ref - 1000) > 0


def test_constant_reference():
    assert psi_calculation([0.5] * 10, [0.5] * 20) == pytest.approx(0)
    assert psi_calculation([0.5] * 10, [0.6] * 10) > 0
    assert psi_calculation([0.5] * 10, [0.4] * 10) > 0


@pytest.mark.parametrize("bins", [2, 10])
def test_binary_distribution_matches_expected_psi(bins):
    ref = [0, 0, 1, 1]
    new = [0, 1, 1, 1]
    expected = np.sum(
        (np.array([0.5, 0.5]) - [0.25, 0.75])
        * np.log(np.array([0.5, 0.5]) / [0.25, 0.75])
    )
    assert psi_calculation(ref, new, bins=bins) == pytest.approx(expected)


@pytest.mark.parametrize("bins", [0, 1, -1, 2.5, True, "10", None])
def test_invalid_bins_are_rejected(bins):
    with pytest.raises(ValueError, match="bins"):
        psi_calculation([0, 1], [0, 1], bins=bins)


@pytest.mark.parametrize("invalid", [[], [np.nan], [np.inf], [-np.inf], [[0, 1]], 1])
@pytest.mark.parametrize("side", ["ref", "new"])
def test_invalid_samples_are_rejected(invalid, side):
    samples = {"ref": [0, 1], "new": [0, 1], side: invalid}
    with pytest.raises(ValueError):
        psi_calculation(**samples)
