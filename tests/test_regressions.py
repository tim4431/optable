"""Regression tests for fixed library bugs."""

import numpy as np

from optable import Monitor, OpticalTable, Polygon, Ray


def test_polygon_within_boundary_numpy2():
    # np.cross on 2-element vectors was removed in numpy 2.0; the
    # point-on-edge check in Polygon.within_boundary must not rely on it.
    tri = Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])  # (y, z) verts, x=0 plane
    assert tri.within_boundary(np.array([0.0, 0.5, 0.5]))  # interior
    assert tri.within_boundary(np.array([0.0, 1.0, 0.0]))  # on edge
    assert not tri.within_boundary(np.array([0.0, 2.0, 2.0]))  # outside


def test_monitor_get_beam_waist():
    # get_beam_waist referenced the nonexistent attribute self.rList
    table = OpticalTable()
    monitor = Monitor([5, 0, 0], width=2.0, height=2.0)
    table.add_monitors(monitor)

    w0 = 1e-3
    wl = 780e-9
    table.ray_tracing(Ray([0, 0, 0], [1, 0, 0], wavelength=wl, w0=w0))

    waists = monitor.get_beam_waist()
    assert len(waists) == 1
    # free-space propagation does not change the waist radius
    assert np.isclose(waists[0], w0)
