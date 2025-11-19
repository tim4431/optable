import numpy as np
from typing import List, Tuple, Union, Sequence
import copy
from dataclasses import dataclass
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

    def _vector_to_R(self, t: np.ndarray) -> np.ndarray:
        """Return 3×3 rotation matrix taking (1,0,0) to t = (tx,ty,tz)."""
        v = self._normalize_vector(t)

        # Special cases: colinear with ±x
        if np.allclose(v, [1, 0, 0]):  # already aligned
            return np.eye(3)
        if np.allclose(v, [-1, 0, 0]):  # opposite; rotate 180° about y-axis
            return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        k = np.cross([1, 0, 0], v)
        k = k / np.linalg.norm(k)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        c = v[0]  # cos_theta
        R = c * np.eye(3) + (1 - c) * np.outer(k, k) + np.sin(np.arccos(c)) * K
        return R

    def _RotAroundLocal(self, axis, localpoint, theta):
        # return self
        raise NotImplementedError("_RotAroundLocal method not implemented")

    def _RotAroundCenter(self, axis, theta):
        return self._RotAroundLocal(axis, [0, 0, 0], theta)

    def RotX(self, theta):
        return self._RotAroundCenter([1, 0, 0], theta)

    def RotY(self, theta):
        return self._RotAroundCenter([0, 1, 0], theta)

    def RotZ(self, theta):
        return self._RotAroundCenter([0, 0, 1], theta)

    def _RotAround(self, axis, point, theta):
        localpoint = np.array(point) - self.origin
        return self._RotAroundLocal(axis, localpoint, theta)

    def RotXAroundLocal(self, localpoint, theta):
        return self._RotAroundLocal([1, 0, 0], localpoint, theta)

    def RotYAroundLocal(self, localpoint, theta):
        return self._RotAroundLocal([0, 1, 0], localpoint, theta)

    def RotZAroundLocal(self, localpoint, theta):
        return self._RotAroundLocal([0, 0, 1], localpoint, theta)

    def _Translate(self, movement):
        self.origin += np.array(movement)
        return self

    def TX(self, dx):
        return self._Translate([dx, 0, 0])

    def TY(self, dy):
        return self._Translate([0, dy, 0])

    def TZ(self, dz):
        return self._Translate([0, 0, dz])


class Path:
    def __init__(self, pts: List):
        self.pts = [np.array(pt) for pt in pts]
        self.pts.append(self.pts[0])  # close the path
        self.accumulated_length = self._accumulate_length()
        self.round_trip = self.accumulated_length[-1]

    def _accumulate_length(self):
        accum_length = [0]
        length = 0
        for i in range(1, len(self.pts)):
            length += np.linalg.norm(np.array(self.pts[i]) - np.array(self.pts[i - 1]))
            accum_length.append(length)
        return accum_length

    def _get_segment(self, l):
        """get the segment index at the position l, from pts[i-1] to pts[i]"""
        # map l into the range of round_trip
        l = l % self.round_trip
        # print(l)
        for i in range(1, len(self.pts)):
            if l <= self.accumulated_length[i]:
                break
        return i

    def coord(self, l):
        """calculate the coordinate at the position l"""
        # calculate the coordinate
        i = self._get_segment(l)
        l0 = self.accumulated_length[i - 1]
        l1 = self.accumulated_length[i]
        # print(i)
        ratio = (l - l0) / (l1 - l0)
        pt0 = np.array(self.pts[i - 1])
        # print(pt0)
        pt1 = np.array(self.pts[i])
        # print(pt1)
        # print(ratio)
        return pt0 + (pt1 - pt0) * ratio

    def direction(self, l):
        i = self._get_segment(l)
        pt0 = np.array(self.pts[i - 1])
        pt1 = np.array(self.pts[i])
        return np.linalg.norm(pt1 - pt0)

    def rotz_theta(self, l):
        direction = self.direction(l)
        return np.arctan2(direction[1], direction[0])

    def bbox(self):
        x = [pt[0] for pt in self.pts]
        y = [pt[1] for pt in self.pts]
        return (min(x) - 0.3, max(x) + 0.3, min(y) - 0.3, max(y) + 0.3)


@dataclass(frozen=True)
class Color:
    SCIENCE_RED_LIGHT: str = "#febfbe"
    SCIENCE_RED_DARK: str = "#fa331a"
    SCIENCE_BLUE_LIGHT: str = "#bdd3ec"
    SCIENCE_BLUE_DARK: str = "#2556ae"


def run_code_block(filepath, marker, globals=None):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_marker = "# >>> " + marker
    end_marker = "# <<< " + marker
    inside_block = False
    code_lines = []
    for line in lines:
        if line.strip() == start_marker:
            inside_block = True
            continue
        if line.strip() == end_marker:
            break
        if inside_block:
            code_lines.append(line)

    print(
        f"Loading code block {len(code_lines)} lines from {filepath} between markers: {start_marker} and {end_marker}"
    )

    code = "".join(code_lines)
    exec(code, globals)


def to_mathematical_str(s):
    if s == "None":
        return "None"
    return s.replace("[", "{").replace("]", "}").replace("e", "*10^").replace("j", "I")


def get_attr_str(obj, attr_name, default=None):
    return (
        default
        if not hasattr(obj, attr_name) or not getattr(obj, attr_name)
        else getattr(obj, attr_name)
    )


def solve_crosssection_ray_bbox(bbox, ray_origin, ray_direction) -> Tuple[float]:
    """for each set of bbox plane, solve the intersection with the ray [t1,t2]
    then the intersection point should be intersection of [t1x,t2x], [t1y,t2y], [t1z,t2z]
    return [t1, t2], if no intersection return [0, inf]
    """
    t1, t2 = 0, np.inf
    for i in range(3):
        if abs(ray_direction[i]) < 1e-12:
            # parallel
            if not (bbox[2 * i] <= ray_origin[i] <= bbox[2 * i + 1]):
                # outside the bbox
                return 0, np.inf
        else:
            t1i = (bbox[2 * i] - ray_origin[i]) / ray_direction[i]
            t2i = (bbox[2 * i + 1] - ray_origin[i]) / ray_direction[i]
            if t1i > t2i:
                t1i, t2i = t2i, t1i

            t1 = max(t1, t1i)
            t2 = min(t2, t2i)
    #
    return t1, t2
