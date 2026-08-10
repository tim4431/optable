"""Geometry primitives: rotation matrices, translations, id-preserving copies."""

import numpy as np

from optable import Ray, Vector


def test_rotation_matrix_is_orthonormal():
    v = Vector([0, 0, 0])
    R = v.R([0, 0, 1], 0.3)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_matrix_rotates_x_to_y():
    v = Vector([0, 0, 0])
    R = v.R([0, 0, 1], np.pi / 2)
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0])


def test_ray_rotz_rotates_direction():
    ray = Ray([0, 0, 0], [1, 0, 0])
    ray.RotZ(np.pi / 2)
    assert np.allclose(ray.direction, [0, 1, 0])


def test_translate():
    v = Vector([1.0, 2.0, 3.0])
    v.TX(1.0).TY(-2.0).TZ(0.5)
    assert np.allclose(v.origin, [2.0, 0.0, 3.5])


def test_copy_preserves_id_and_is_independent():
    ray = Ray([0, 0, 0], [1, 0, 0])
    clone = ray.copy()
    assert clone._id == ray._id
    clone.origin[0] = 42.0
    assert ray.origin[0] == 0.0


def test_ray_direction_is_normalized():
    ray = Ray([0, 0, 0], [3, 4, 0])
    assert np.isclose(np.linalg.norm(ray.direction), 1.0)
    assert np.allclose(ray.direction, [0.6, 0.8, 0])
