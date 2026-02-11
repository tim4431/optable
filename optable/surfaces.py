from .base import *
from .solver import *


class Surface(Base):
    def __init__(self):
        """
        local coordinate system
        """
        super().__init__()
        self.planar = True  # True if the surface is planar

    def _normalize_vector(self, vector) -> np.ndarray:
        vec = np.array(vector, dtype=float)
        return vec / np.linalg.norm(vec)

    def f(self, P: np.ndarray) -> float:
        """f(x,y,z) = 0"""
        raise NotImplementedError(
            "Method 'f' must be implemented in the derived class."
        )

    def normal(self, P: np.ndarray) -> np.ndarray:
        """Return the normal vector at the point."""
        raise NotImplementedError(
            "Method 'normal' must be implemented in the derived class."
        )

    def within_boundary(self, P: np.ndarray) -> bool:
        """Check if the point is inside the boundary."""
        raise NotImplementedError(
            "Method 'within_boundary' must be implemented in the derived class."
        )

    def parametric_boundary_3d(self, t: float) -> np.ndarray:
        """Return the point on the boundary given the parameter t=[0,1]."""
        raise NotImplementedError(
            "Method 'parametric_boundary' must be implemented in the derived class"
        )

    def merge_bbox(self, bbox1, bbox2):
        return (
            min(bbox1[0], bbox2[0]),
            max(bbox1[1], bbox2[1]),
            min(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3]),
            min(bbox1[4], bbox2[4]),
            max(bbox1[5], bbox2[5]),
        )

    def merge_bboxs(self, bboxs):
        return (
            np.min([b[0] for b in bboxs]),
            np.max([b[1] for b in bboxs]),
            np.min([b[2] for b in bboxs]),
            np.max([b[3] for b in bboxs]),
            np.min([b[4] for b in bboxs]),
            np.max([b[5] for b in bboxs]),
        )

    def solve_crosssection_ray_bbox_local(
        self, ray_origin, ray_direction
    ) -> Tuple[float]:
        bbox_local = self.get_bbox_local()
        return solve_ray_bboxes_intersections(ray_origin, ray_direction, bbox_local)


class Point(Surface):
    def __init__(self):
        super().__init__()
        self.planar = False

    def f(self, P: np.ndarray) -> float:
        return np.linalg.norm(P)

    def normal(self, P: np.ndarray) -> np.ndarray:
        return P / np.linalg.norm(P)

    def within_boundary(self, P: np.ndarray) -> bool:
        return np.linalg.norm(P) < 1e-12  # only the origin

    def parametric_boundary(self, t: Sequence[float], type: str) -> np.ndarray:
        return np.zeros((3, len(t)))

    def get_bbox_local(self):
        return (0, 0, 0, 0, 0, 0)


class Plane(Surface):
    def __init__(self):
        super().__init__()
        self._normal = np.array([1, 0, 0])

    def f(self, P: np.ndarray) -> float:
        return np.dot(self._normal, P)

    def normal(self, P: np.ndarray) -> np.ndarray:
        return self._normal

    def union(self, other):
        obj = Plane()

        def within_boundary(P):
            return self.within_boundary(P) or other.within_boundary(P)

        def parametric_boundary(t, type):
            return np.hstack(
                [self.parametric_boundary(t, type), other.parametric_boundary(t, type)]
            )

        def get_bbox_local():
            return self.merge_bbox(self.get_bbox_local(), other.get_bbox_local())

        obj.within_boundary = within_boundary
        obj.parametric_boundary = parametric_boundary
        obj.get_bbox_local = get_bbox_local
        return obj

    def subtract(self, other):
        obj = Plane()

        def within_boundary(P):
            return self.within_boundary(P) and not other.within_boundary(P)

        def parametric_boundary(t, type):
            return np.hstack(
                [self.parametric_boundary(t, type), other.parametric_boundary(t, type)]
            )

        def get_bbox_local():
            return self.merge_bbox(self.get_bbox_local(), other.get_bbox_local())

        obj.within_boundary = within_boundary
        obj.parametric_boundary = parametric_boundary
        obj.get_bbox_local = get_bbox_local
        return obj


class Circle(Plane):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def within_boundary(self, P: np.ndarray) -> bool:
        return np.linalg.norm(P) <= self.radius

    def parametric_boundary(self, t: Sequence[float], type: str) -> np.ndarray:
        if type == "Z":
            x = np.array([0, 0])
            y = np.array([-self.radius, self.radius])
            z = np.array([0, 0])
        elif type == "3D":
            theta = 2 * np.pi * t
            x = np.zeros_like(theta)
            y = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
        points = np.vstack([x, y, z])
        return points

    def get_bbox_local(self):
        return (0, 0, -self.radius, self.radius, -self.radius, self.radius)


class Rectangle(Plane):
    def __init__(self, width, height):
        super().__init__()
        self.width = width  # y
        self.height = height  # z

    def within_boundary(self, P: np.ndarray) -> bool:
        return np.abs(P[1]) <= self.width / 2 and np.abs(P[2]) <= self.height / 2

    def parametric_boundary(self, t: Sequence[float], type: str) -> np.ndarray:
        if type == "Z":
            x = np.array([0, 0])
            y = np.array([-self.width / 2, self.width / 2])
            z = np.array([0, 0])
        elif type == "3D":
            x = np.zeros(5)
            y = np.array(
                [
                    -self.width / 2,
                    self.width / 2,
                    self.width / 2,
                    -self.width / 2,
                    -self.width / 2,
                ]
            )
            z = np.array(
                [
                    -self.height / 2,
                    -self.height / 2,
                    self.height / 2,
                    self.height / 2,
                    -self.height / 2,
                ]
            )
        points = np.vstack([x, y, z])
        return points

    def get_bbox_local(self):
        return (
            0,
            0,
            -self.width / 2,
            self.width / 2,
            -self.height / 2,
            self.height / 2,
        )


class Cylinder(Surface):
    def __init__(self, radius, height, theta_range=(-np.pi, np.pi)):
        super().__init__()
        self.radius = radius
        self.height = height
        self.theta_range = theta_range
        self.planar = False

    def _theta(self, t):
        return self.theta_range[0] + (self.theta_range[1] - self.theta_range[0]) * t

    def f(self, P: np.ndarray) -> float:
        return np.linalg.norm(P[:2]) - self.radius

    def normal(self, P: np.ndarray) -> np.ndarray:
        return np.array([P[0], P[1], 0]) / self.radius

    def within_boundary(self, P: np.ndarray) -> bool:
        z = P[2]
        theta = np.arctan2(P[1], P[0])  # range: [-pi, pi]
        return (self.theta_range[0] <= theta <= self.theta_range[1]) and (
            -self.height / 2 <= z <= self.height / 2
        )

    def parametric_boundary(self, t: Sequence[float], type: str) -> np.ndarray:
        if type == "Z":
            theta = self._theta(t)
            x = self.radius * np.cos(theta)
            y = self.radius * np.sin(theta)
            z = np.zeros_like(t)
        elif type == "3D":
            x = np.zeros_like(t)
            y = np.zeros_like(t)
            z = np.zeros_like(t)
            # 0<=t<=1/4, top circle, map t to theta_range
            t0_mask = (0 <= t) & (t < 1 / 4)
            theta = self._theta((t[t0_mask] - 1 / 4) * 4)
            x[t0_mask] = self.radius * np.cos(theta)
            y[t0_mask] = self.radius * np.sin(theta)
            z[t0_mask] = self.height / 2
            # 1/4<=t<=1/2, right side
            t1_mask = (1 / 4 <= t) & (t < 1 / 2)
            theta_max = self.theta_range[1]
            x[t1_mask] = self.radius * np.cos(theta_max)
            y[t1_mask] = self.radius * np.sin(theta_max)
            z[t1_mask] = self.height / 2 - (t[t1_mask] - 1 / 4) * self.height * 2
            # 1/2<=t<=3/4, bottom circle
            t2_mask = (1 / 2 <= t) & (t < 3 / 4)
            theta = self._theta((t[t2_mask] - 1 / 2) * 4)
            x[t2_mask] = self.radius * np.cos(theta)
            y[t2_mask] = self.radius * np.sin(theta)
            z[t2_mask] = -self.height / 2
            # 3/4<=t<=1, left side
            t3_mask = (3 / 4 <= t) & (t <= 1)
            theta_min = self.theta_range[0]
            x[t3_mask] = self.radius * np.cos(theta_min)
            y[t3_mask] = self.radius * np.sin(theta_min)
            z[t3_mask] = -self.height / 2 + (t[t3_mask] - 3 / 4) * self.height * 2
        points = np.vstack([x, y, z])
        return points

    def get_bbox_local(self):
        return (
            -self.radius,
            self.radius,
            -self.radius,
            self.radius,
            -self.height / 2,
            self.height / 2,
        )


class Sphere(Surface):
    def __init__(self, radius, height=None):
        super().__init__()
        self.radius = radius
        self.height = (
            height if height is not None else 2 * radius
        )  # from z=+radius to z=+radius-height
        self.planar = False
        self.diameter = np.sqrt(radius**2 - (radius - height) ** 2) * 2

    def f(self, P: np.ndarray) -> float:
        return np.linalg.norm(P) - self.radius

    def normal(self, P: np.ndarray) -> np.ndarray:
        return P / self.radius

    def within_boundary(self, P: np.ndarray) -> bool:
        EPS = 1e-12
        x, y, z = P
        return self.radius - self.height - EPS <= x and x <= self.radius + EPS

    def parametric_boundary(self, t: Sequence[float], type: str) -> np.ndarray:
        theta = 2 * np.pi * t
        # bottom circle
        r = np.sqrt(self.radius**2 - (self.radius - self.height) ** 2)
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        x = np.full_like(t, self.radius - self.height)
        points = np.vstack([x, y, z])
        # curved top 1
        theta_r = np.arccos((self.radius - self.height) / self.radius)
        theta = (2 * theta_r) * (t - 0.5)
        x = self.radius * np.cos(theta)
        z = self.radius * np.sin(theta)
        y = np.full_like(t, 0.0)
        points = np.hstack([points, np.vstack([x, y, z])])
        # curved top 2
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        z = np.full_like(t, 0.0)
        points = np.hstack([points, np.vstack([x, y, z])])

        return points

    def get_bbox_local(self):
        return (
            self.radius - self.height,
            self.radius,
            -self.diameter / 2,
            self.diameter / 2,
            -self.diameter / 2,
            self.diameter / 2,
        )


class ASphere(Surface):
    def __init__(self, radius, f_asphere: callable):
        """x = -f_asphere(r)"""
        super().__init__()
        self.radius = radius
        self.f_asphere = f_asphere
        self.planar = False
        x0 = -self.f_asphere(0)
        xR = -self.f_asphere(self.radius)
        self.xmin = min(x0, xR)
        self.xmax = max(x0, xR)

    def _df_asphere_dr(self, r: float) -> float:
        h = 1e-4 * self.radius
        return (self.f_asphere(r + h) - self.f_asphere(r - h)) / (2 * h)

    def _df_asphere_dr2(self, r: float) -> float:
        # use central difference to compute second derivative
        h = 1e-4 * self.radius
        return (
            self.f_asphere(r + h) - 2 * self.f_asphere(r) + self.f_asphere(r - h)
        ) / (h**2)

    def roc_r(self, r: float) -> float:
        """Return the radius of curvature at radius r."""
        df_dr = self._df_asphere_dr(r)
        df_dr2 = self._df_asphere_dr2(r)
        roc = (
            1 + df_dr**2
        ) ** 1.5 / df_dr2  # roc>0 means center of curvature toward -x side
        return roc

    def roc(self, P: np.ndarray) -> float:
        r = np.linalg.norm(P[1:3])
        return self.roc_r(r)

    def f(self, P: np.ndarray) -> float:
        r = np.linalg.norm(P[1:3])
        z_asphere = -self.f_asphere(r)
        return P[0] - z_asphere

    def normal(self, P: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(P[1:3])
        if r < 1e-12:
            n = np.array([1.0, 0.0, 0.0])
        else:
            df_dr = self._df_asphere_dr(r)
            n = np.array([1, (df_dr) * (P[1] / r), (df_dr) * (P[2] / r)])
            n = n / np.linalg.norm(n)
        return n

    def within_boundary(self, P: np.ndarray) -> bool:
        EPS = 1e-12
        r = np.linalg.norm(P[1:3])
        return r <= self.radius + EPS

    def parametric_boundary(self, t: Sequence[float], type: str) -> np.ndarray:
        theta = 2 * np.pi * t
        # outer circle
        r = self.radius
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        x = -self.f_asphere(r) * np.ones_like(t)
        points = np.vstack([x, y, z])
        # curved top 1
        r = np.linspace(-self.radius, self.radius, len(t))
        x = -self.f_asphere(np.abs(r))
        z = np.zeros_like(t)
        y = r * np.full_like(t, 1.0)
        points = np.hstack([points, np.vstack([x, y, z])])
        # curved top 2
        y = r * np.zeros_like(t)
        z = r * np.full_like(t, 1.0)
        points = np.hstack([points, np.vstack([x, y, z])])
        return points

    def get_bbox_local(self):
        return (
            self.xmin,
            self.xmax,
            -self.radius,
            self.radius,
            -self.radius,
            self.radius,
        )


class Polygon(Plane):
    """
    Planar polygon surface.

    *If* vertices are supplied as (N, 2) or as (N, 3) with every x = 0,
    the polygon is taken to lie in the x = 0 plane with normal (1, 0, 0).

    Parameters
    ----------
    vertices : Sequence[Sequence[float]]
        Coordinates of the polygon vertices (N ≥ 3, 2- or 3-D).
        Ordering **must** be counter-clockwise when viewed *along* the
        supplied/implicit normal.
    normal : Sequence[float] | None
        Optional outward normal.  Required if the vertices are not in a single
        x = const plane.
    """

    def __init__(
        self,
        vertices: Sequence[Sequence[float]],
        normal: Sequence[float] | None = None,
    ):
        super().__init__()  # Plane sets self._normal
        self._tol = 1e-9

        verts = np.asarray(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[0] < 3:
            raise ValueError("Need at least three vertices (shape (N,2) or (N,3)).")

        self.planar = False  # planar polygon in x = 0 plane
        # ---------------------------------------------------------------
        # 1.  Promote 2-D vertices → 3-D in the x = 0 plane
        # ---------------------------------------------------------------
        if verts.shape[1] == 2:  # (y, z) given
            verts = np.column_stack((np.zeros(len(verts)), verts))
            if normal is None:
                normal = (1.0, 0.0, 0.0)
                self.planar = True  # planar polygon in x = 0 plane

        # If 3-D vertices are all in x = 0 and no normal was given,
        # assume the default plane too.
        if verts.shape[1] == 3 and normal is None and np.allclose(verts[:, 0], 0.0):
            normal = (1.0, 0.0, 0.0)
            self.planar = True  # planar polygon in x = 0 plane

        self.vertices = verts

        # ---------------------------------------------------------------
        # 2.  Establish or verify the plane normal
        # ---------------------------------------------------------------
        if normal is None:
            # derive from first non-colinear triplet
            found = False
            for i in range(2, len(verts)):
                n = np.cross(verts[i] - verts[0], verts[1] - verts[0])
                if np.linalg.norm(n) > self._tol:
                    normal = n
                    found = True
                    break
            if not found:
                raise ValueError("Vertices are colinear - cannot define a plane.")

        self._normal = self._normalize_vector(normal)

        # Coplanarity check
        if np.any(np.abs((verts - verts[0]) @ self._normal) > self._tol):
            raise ValueError("Vertices are not coplanar with the supplied normal.")

        # ---------------------------------------------------------------
        # 3.  Build an orthonormal (u,v) basis in the plane
        # ---------------------------------------------------------------
        a = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(a, self._normal)) > 0.99:  # almost parallel
            a = np.array([0.0, 1.0, 0.0])
        u = self._normalize_vector(np.cross(self._normal, a))
        v = np.cross(self._normal, u)
        self._basis = (u, v)

        # Pre-project all vertices once
        self._verts2d = self._project_to_2d(verts)

        # Axis-aligned 3-D bounding box (fast rejection)
        self._bbox = (
            verts[:, 0].min(),
            verts[:, 0].max(),
            verts[:, 1].min(),
            verts[:, 1].max(),
            verts[:, 2].min(),
            verts[:, 2].max(),
        )

    # ------------------------------------------------------------------ #
    #                    Internal helpers                                 #
    # ------------------------------------------------------------------ #
    def _project_to_2d(self, pts: np.ndarray) -> np.ndarray:
        """Project 3-D points into the polygon's local (u,v) coordinates."""
        u, v = self._basis
        d = pts - self.vertices[0]
        return np.column_stack((d @ u, d @ v))

    # ------------------------------------------------------------------ #
    #             Plane/Surface interface overrides                       #
    # ------------------------------------------------------------------ #
    def f(self, P: np.ndarray) -> float:
        """Signed distance from point to the polygon's plane."""
        return np.dot(self._normal, P - self.vertices[0])

    def within_boundary(self, P: np.ndarray) -> bool:
        # The within boundary criterion is determined by the 2-D even–odd ray-casting in (u,v) plane
        px, py = self._project_to_2d(P[None, :])[0]
        verts = self._verts2d
        inside = False

        for i in range(len(verts)):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % len(verts)]

            # (a) exactly on edge?  treat as inside
            if (
                np.abs(np.cross([x2 - x1, y2 - y1], [px - x1, py - y1])) <= self._tol
                and min(x1, x2) - self._tol <= px <= max(x1, x2) + self._tol
                and min(y1, y2) - self._tol <= py <= max(y1, y2) + self._tol
            ):
                return True

            # (b) even-odd rule
            if (y1 > py) != (y2 > py):
                x_at_y = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if x_at_y >= px:
                    inside = not inside

        return inside

    # Optional: provide the boundary path (only type == "3D" is implemented)
    def parametric_boundary(self, t: Sequence[float], type: str) -> np.ndarray:
        # if type != "3D":
        #     raise NotImplementedError("Only '3D' parametric boundary is supported.")
        verts_closed = np.vstack([self.vertices, self.vertices[0]])  # close the loop
        return verts_closed.T

    def get_bbox_local(self) -> Tuple[float, float, float, float, float, float]:
        return self._bbox
