"""End-to-end ray tracing: reflection, thin-lens focusing, monitors, ABCD."""

import numpy as np

from optable import Lens, Mirror, Monitor, OpticalTable, Ray


def test_mirror_reflects_45_degrees():
    table = OpticalTable()
    mirror = Mirror([10, 0, 0], radius=1.0)
    mirror.RotZ(3 * np.pi / 4)  # surface normal faces the incoming +X ray
    table.add_components(mirror)

    source = Ray([0, 0, 0], [1, 0, 0])
    traced = table.ray_tracing(source)

    reflected = [r for r in traced if np.allclose(r.direction, [0, 1, 0])]
    assert len(reflected) == 1
    assert np.allclose(reflected[0].origin, [10, 0, 0])
    # ray identity is preserved across interactions
    assert reflected[0]._id == source._id


def test_thin_lens_focuses_parallel_rays():
    focal_length = 5.0
    table = OpticalTable()
    table.add_components(Lens([10, 0, 0], focal_length=focal_length, radius=1.0))
    focal_plane = Monitor([10 + focal_length, 0, 0], width=2.0, height=2.0)
    table.add_monitors(focal_plane)

    rays = [Ray([0, y, 0], [1, 0, 0]) for y in (-0.2, 0.1, 0.3)]
    table.ray_tracing(rays)

    assert focal_plane.ndata == 3
    assert np.allclose(focal_plane.yList, 0.0, atol=1e-9)


def test_monitor_records_intensity():
    table = OpticalTable()
    monitor = Monitor([5, 0, 0], width=2.0, height=2.0)
    table.add_monitors(monitor)

    table.ray_tracing(Ray([0, 0.5, 0], [1, 0, 0], intensity=0.25))

    assert monitor.ndata == 1
    assert np.isclose(monitor.sum_intensity, 0.25)
    assert np.isclose(monitor.yList[0], 0.5)


def test_free_space_abcd_matrix():
    d = 5.0
    table = OpticalTable()
    mon0 = Monitor([5, 0, 0], width=2.0, height=2.0)
    mon1 = Monitor([5 + d, 0, 0], width=2.0, height=2.0)
    table.add_monitors([mon0, mon1])

    rays = [Ray([0, 0, 0], [1, 0, 0])]
    Ms = table.calculate_abcd_matrix(mon0, mon1, rays)

    assert Ms.shape == (1, 2, 2)
    assert np.allclose(Ms[0], [[1.0, d], [0.0, 1.0]], atol=1e-4)


def test_facing_mirrors_terminate_via_performance_limit():
    table = OpticalTable()
    m1 = Mirror([0, 0, 0], radius=1.0)
    m2 = Mirror([10, 0, 0], radius=1.0)
    m2.RotZ(np.pi)  # normals face each other -> infinite cavity bounces
    table.add_components([m1, m2])

    traced = table.ray_tracing(
        Ray([5, 0, 0], [1, 0, 0]), perfomance_limit={"max_trace_num": 20}
    )
    # must return instead of bouncing forever
    assert isinstance(traced, list)
