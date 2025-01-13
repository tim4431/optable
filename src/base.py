import numpy as np
from typing import List, Tuple, Union
import copy
import matplotlib.pyplot as plt


class Base:
    def __init__(self, **kwargs):
        if not hasattr(self, "_id"):
            self._id = id(self)  # when a ray copy is made, the id will be inherited
        for key, value in kwargs.items():
            setattr(self, key, value)

    def copy(self, **kwargs):
        obj = copy.deepcopy(self)
        for key, value in kwargs.items():
            setattr(obj, key, value)
        return obj


class Vector(Base):
    def __init__(self, origin, **kwargs):
        super().__init__(**kwargs)
        self.origin = np.array(origin, dtype=float)

    def _normalize_vector(self, vector) -> np.ndarray:
        vec = np.array(vector, dtype=float)
        return vec / np.linalg.norm(vec)

    def R(self, axis, theta: float) -> np.ndarray:
        # u is the rotaion axis, theta is the rotation angle
        """
        Returns the rotation matrix for a proper rotation by angle theta
        around the axis (u_x, u_y, u_z), where the axis is a unit vector.

        Parameters:
            axis: tuple or list of length 3 (unit vector u_x, u_y, u_z)
            theta: float, rotation angle in radians

        Returns:
            A 3x3 numpy array representing the rotation matrix.
        """
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        u_x, u_y, u_z = axis
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        one_minus_cos = 1 - cos_theta

        # Rotation matrix components, see https://en.wikipedia.org/wiki/Rotation_matrix
        R = np.array(
            [
                [
                    u_x**2 * one_minus_cos + cos_theta,
                    u_x * u_y * one_minus_cos - u_z * sin_theta,
                    u_x * u_z * one_minus_cos + u_y * sin_theta,
                ],
                [
                    u_y * u_x * one_minus_cos + u_z * sin_theta,
                    u_y**2 * one_minus_cos + cos_theta,
                    u_y * u_z * one_minus_cos - u_x * sin_theta,
                ],
                [
                    u_z * u_x * one_minus_cos - u_y * sin_theta,
                    u_z * u_y * one_minus_cos + u_x * sin_theta,
                    u_z**2 * one_minus_cos + cos_theta,
                ],
            ]
        )

        return R

    def _RotAroundCenter(self, axis, theta):
        return self

    def RotX(self, theta):
        return self._RotAroundCenter([1, 0, 0], theta)

    def RotY(self, theta):
        return self._RotAroundCenter([0, 1, 0], theta)

    def RotZ(self, theta):
        return self._RotAroundCenter([0, 0, 1], theta)

    def _RotAroundLocal(self, axis, localpoint, theta):
        return self

    def RotXAroundLocal(self, localpoint, theta):
        return self._RotAroundLocal([1, 0, 0], localpoint, theta)

    def RotYAroundLocal(self, localpoint, theta):
        return self._RotAroundLocal([0, 1, 0], localpoint, theta)

    def RotZAroundLocal(self, localpoint, theta):
        return self._RotAroundLocal([0, 0, 1], localpoint, theta)

    def _Translate(self, direction, distance):
        self.origin += np.array(direction) * distance
        return self

    def TX(self, dx):
        return self._Translate([1, 0, 0], dx)

    def TY(self, dy):
        return self._Translate([0, 1, 0], dy)

    def TZ(self, dz):
        return self._Translate([0, 0, 1], dz)
