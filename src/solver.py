import numpy as np
from typing import List, Tuple, Union


def solve_ray_bboxes_intersections(
    ray_origin, ray_direction, bboxes: Union[List[Tuple], Tuple]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """for each set of bbox plane, solve the intersection with the ray [t1,t2]
    then the intersection point should be intersection of [t1x,t2x], [t1y,t2y], [t1z,t2z]
    return [t1, t2], if no intersection return [0, inf]
    """
    EPS = 1e-12
    if isinstance(bboxes, tuple):
        bboxes = [bboxes]
    bboxes = np.array(bboxes)  # (n_boxes, 6)
    #
    # t1, t2 = 0, np.inf
    t1, t2 = np.full(bboxes.shape[0], 0, dtype=float), np.full(
        bboxes.shape[0], np.inf, dtype=float
    )
    for axis in range(3):
        o = ray_origin[axis]
        d = ray_direction[axis]
        bmin = bboxes[:, 2 * axis]
        bmax = bboxes[:, 2 * axis + 1]
        #
        if np.isclose(d, 0.0):
            # parallel
            outside = (o < bmin) | (o > bmax)
            # Mark no-hit by forcing t1 > t2 for those
            t1[outside] = 1
            t2[outside] = 0
            continue
        inv_d = 1.0 / d
        t_axis0 = (bmin - o) * inv_d
        t_axis1 = (bmax - o) * inv_d

        t_axis_near = np.minimum(t_axis0, t_axis1)
        t_axis_far = np.maximum(t_axis0, t_axis1)

        # Update global t1, t2 using this axis
        t1 = np.maximum(t1, t_axis_near)
        t2 = np.minimum(t2, t_axis_far)

    # Valid hit: intervals overlap and intersection not entirely behind the origin
    hit = (t2 + EPS >= t1) & (t2 >= 0.0)

    return t1, t2, hit


def solve_ray_ray_intersection(
    ray1_origin, ray1_direction, ray2_origin, ray2_direction
):
    """
    Computes ray-ray intersection/closest point with hard clamping for t > 0.

    Returns:
        t1, t2 (float): Parameters for the closest points.
        P (np.array): intersection_point.
        n (np.array): The surface normal that reflects beam1 to beam2.
    """
    # 1. Convert to numpy arrays and Normalize Directions
    # Normalizing is crucial for the reflection normal calculation to be correct.
    p1 = np.array(ray1_origin, dtype=np.float64)
    d1 = np.array(ray1_direction, dtype=np.float64)
    d1 = d1 / np.linalg.norm(d1)
    p2 = np.array(ray2_origin, dtype=np.float64)
    d2 = np.array(ray2_direction, dtype=np.float64)
    d2 = d2 / np.linalg.norm(d2)

    # 2. Least Squares Solution for Infinite Lines
    # We solve the linear system for the segment perpendicular to both lines
    r = p1 - p2

    # Dot products
    a = np.dot(d1, d1)  # 1.0 since normalized
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)  # 1.0 since normalized
    d = np.dot(d1, r)
    e = np.dot(d2, r)

    denom = a * c - b * b

    # Check for parallel rays
    if denom < 1e-6:
        t1 = 0.0
        t2 = e / c  # Projection onto d2
    else:
        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom

    # 3. Hard Clamp (as requested)
    t1 = max(0.0, t1)
    t2 = max(0.0, t2)

    # 4. Calculate Closest Points
    point1 = p1 + t1 * d1
    point2 = p2 + t2 * d2

    # The intersection is the midpoint of the closest approach
    P = (point1 + point2) * 0.5

    # 5. Calculate Reflection Normal
    norm = -(d1 + d2) * 0.5  # both rays point towards (-norm)
    n = norm / np.linalg.norm(norm)

    return t1, t2, P, n


def solve_normal_to_normal_rotation(n1, n2) -> Tuple[np.ndarray, float]:
    """Solve the rotation axis and angle to rotate n1 to n2

    Args:
        n1 (np.ndarray): normal vector 1
        n2 (np.ndarray): normal vector 2

    Returns:
        Tuple[np.ndarray,float]: rotation axis, rotation angle in radian
    """
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    #
    axis = np.cross(n1, n2)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        # parallel
        return np.array([1, 0, 0]), 0.0
    axis = axis / axis_norm
    #
    cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return axis, theta
