"""Gaussian beam q-parameter helpers against textbook formulas."""

import numpy as np

from optable import GaussianBeam

W0 = 1e-3
WL = 780e-9
ZR = np.pi * W0**2 / WL


def test_q_at_waist_is_i_times_rayleigh_range():
    q = GaussianBeam.q_at_waist(W0, WL)
    assert np.isclose(np.real(q), 0.0)
    assert np.isclose(np.imag(q), ZR)


def test_rayleigh_range_roundtrip():
    q = GaussianBeam.q_at_waist(W0, WL)
    assert np.isclose(GaussianBeam.rayleigh_range(q), ZR)


def test_waist_roundtrip():
    q = GaussianBeam.q_at_waist(W0, WL)
    assert np.isclose(GaussianBeam.waist(q, WL), W0)


def test_spot_size_at_waist_and_rayleigh_range():
    q = GaussianBeam.q_at_waist(W0, WL)
    assert np.isclose(GaussianBeam.spot_size(q, 0.0, WL), W0)
    # w(z_R) = w0 * sqrt(2)
    assert np.isclose(GaussianBeam.spot_size(q, ZR, WL), W0 * np.sqrt(2))


def test_radius_of_curvature():
    q = GaussianBeam.q_at_waist(W0, WL)
    z = 0.7 * ZR
    expected = z * (1 + (ZR / z) ** 2)
    assert np.isclose(GaussianBeam.radius_of_curvature(q + z), expected)


def test_propagation_shifts_waist_distance():
    q = GaussianBeam.q_at_waist(W0, WL)
    q2 = GaussianBeam.q_at_z(q, 1.25)
    assert np.isclose(GaussianBeam.distance_to_waist(q2), 1.25)
