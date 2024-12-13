from base import *


class Surface(Base):
    def __init__(self):
        """
        local coordinate system
        """
        super().__init__()
        self.planar = True  # True if the surface is planar

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


class Plane(Surface):
    def __init__(self):
        super().__init__()
        self._normal = np.array([1, 0, 0])

    def f(self, P: np.ndarray) -> float:
        return np.dot(self._normal, P)

    def normal(self, P: np.ndarray) -> np.ndarray:
        return self._normal


class Circle(Plane):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def within_boundary(self, P: np.ndarray) -> bool:
        return np.linalg.norm(P) <= self.radius

    def parametric_boundary(self, t: float, type: str) -> np.ndarray:
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

    def get_bbox(self):
        return (0, 0, -self.radius, self.radius, -self.radius, self.radius)


class Rectangle(Plane):
    def __init__(self, width, height):
        super().__init__()
        self.width = width  # y
        self.height = height  # z

    def within_boundary(self, P: np.ndarray) -> bool:
        return np.abs(P[1]) <= self.width / 2 and np.abs(P[2]) <= self.height / 2

    def parametric_boundary(self, t: float, type: str) -> np.ndarray:
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

    def get_bbox(self):
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

    def parametric_boundary(self, t: float, type: str) -> np.ndarray:
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

    def get_bbox(self):
        return (
            -self.radius,
            self.radius,
            -self.radius,
            self.radius,
            -self.height / 2,
            self.height / 2,
        )
