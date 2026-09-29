"""Independent physical limits and field-integral checks for the cat's eye."""

from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.integrate import quad

from optable import gaussian_mode_overlap


CatEyeDesign = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples" / "cat_eye_reflector.py")
)["CatEyeDesign"]


def test_nominal_roundtrip_and_rear_waist():
    design = CatEyeDesign()
    r = design.simulate()
    assert design.index == pytest.approx(1.4481868466)
    assert design.thickness == pytest.approx(.01670743167)
    assert abs(r["q_mirror"].real) < 1e-15
    assert r["q_return"] == pytest.approx(1j * design.rayleigh_range)
    assert r["overlap"] == pytest.approx(1, abs=1e-13)
    assert r["mirror_radius"] == pytest.approx(19.80057129e-6)


def test_equal_waist_displacement_and_tilt_limits():
    w, wavelength = 200e-6, 1188e-9
    q = 1j * np.pi*w*w/wavelength
    d, a = 80e-6, .6e-3
    expected = np.exp(-(d/w)**2 - (np.pi*w*a/wavelength)**2)
    assert gaussian_mode_overlap(q, q, wavelength, (d, 0), (a, 0)) == pytest.approx(expected)


def test_overlap_against_numerical_complex_field_integral():
    wavelength = 1188e-9
    q1, q2 = .04+.11j, -.025+.075j
    delta, angle = 70e-6, .7e-3
    k = 2*np.pi/wavelength
    def field(x, q, d=0, theta=0):
        a = .5j*k/q
        return (2*a.real/np.pi)**.25 * np.exp(-a*(x-d)**2-1j*k*theta*x)
    def integral(d, theta):
        fn = lambda x: np.conj(field(x, q1))*field(x, q2, d, theta)
        return quad(lambda x: fn(x).real, -.004, .004, epsabs=1e-11)[0] + 1j*quad(
            lambda x: fn(x).imag, -.004, .004, epsabs=1e-11)[0]
    expected = abs(integral(delta, angle)*integral(0, 0))**2
    assert gaussian_mode_overlap(q1, q2, wavelength, (delta, 0), (angle, 0)) == pytest.approx(expected)


@pytest.mark.parametrize("params", [dict(decenter=1e-6), dict(tilt=1e-5),
                                   dict(decenter=1e-6, tilt=-1e-5, axial=.001)])
def test_exact_snell_chief_ray_agrees_at_small_misalignment(params):
    design = CatEyeDesign()
    r, ray = design.simulate(**params), design.trace_ray(**params)
    exact_angle = (ray[-1, 1]-ray[-2, 1])/(ray[-2, 0]-ray[-1, 0])
    assert ray[-1, 1] == pytest.approx(r["displacement"], rel=1e-5, abs=1e-10)
    assert exact_angle == pytest.approx(r["angle"], rel=1e-5, abs=1e-10)


def test_focal_plane_limit_is_direction_retroreflecting():
    design = CatEyeDesign()
    delta_t = design.index / design.power - design.thickness
    r = design.simulate(decenter=30e-6, tilt=.002, thickness_error=delta_t)
    assert abs(r["angle"]) < 1e-15
    assert r["displacement"] == pytest.approx(2*30e-6-2*.002/design.power)
    # Geometric focal plane is not the Gaussian waist for this finite input q.
    assert r["overlap"] < .999


def test_fixed_optic_translation_and_symmetry():
    design = CatEyeDesign()
    assert design.simulate(axial=.01)["thickness"] == design.thickness
    assert design.simulate(axial=.01)["overlap"] < 1
    p = design.simulate(decenter=80e-6, tilt=.003)
    m = design.simulate(decenter=-80e-6, tilt=-.003)
    assert p["overlap"] == pytest.approx(m["overlap"])
    assert p["displacement"] == pytest.approx(-m["displacement"])


def test_missed_ray_and_invalid_geometry():
    design = CatEyeDesign()
    assert design.trace_ray(decenter=.01) is None
    with pytest.raises(ValueError):
        design.simulate(axial=-.06)
    with pytest.raises(ValueError):
        CatEyeDesign(aperture_radius=.006)
