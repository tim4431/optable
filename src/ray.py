from .base import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_RAY_NONE_LENGTH = 50


class GaussianBeam:
    @staticmethod
    def q_at_waist(w0, wl, n=1):
        return (1j * n * np.pi * w0**2) / wl

    @staticmethod
    def q_at_z(qo, z):
        return qo + z

    @staticmethod
    def distance_to_waist(q):
        z = np.real(q)
        return z

    @staticmethod
    def waist(q, wl, n=1):
        w0 = np.sqrt((wl * np.imag(q)) / (n * np.pi))
        return float(w0)

    @staticmethod
    def rayleigh_range(q):
        zr = np.imag(q)
        return float(zr)

    @staticmethod
    def radius_of_curvature(q):
        R = 1 / np.real(1 / q)
        return float(R)

    @staticmethod
    def spot_size(qo, z, wl, n=1):
        q = qo + z
        w = np.sqrt(-wl / (n * np.pi * np.imag(1 / q)))
        return w


class Ray(Vector):
    def __init__(
        self,
        origin,
        direction,
        intensity: float = 1.0,
        wavelength=None,
        length=None,
        alive=True,
        qo=None,  # q at origin
        w0=None,  # if specified, qo will be calculated from w0
        **kwargs,
    ):
        super().__init__(origin, **kwargs)
        self.length = float(length) if length else None  # Length of the ray
        self.direction = direction  # Direction vector
        self.intensity = float(intensity)  # Intensity of the ray
        self.wavelength = (
            float(wavelength) if wavelength else None
        )  # Wavelength of the ray
        self.alive = alive  # Ray is alive means it has not been absorbed or terminated
        self.n = 1.0  # refractive index, default is 1.0
        #
        # >>> Gaussian Beam Parameters
        if qo is not None:
            self.qo = qo  # q at origin
        elif w0 is not None:
            self.qo = self.q_at_waist(w0)
        else:
            self.qo = None
        #

    def __repr__(self):
        return f"Ray(origin={self.origin}, direction={self.direction}, intensity={self.intensity}, length={self.length}, alive={self.alive}, qo={self.qo})"

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        self._direction = self._normalize_vector(direction)
        # print(self._direction)

    @property
    def transform_matrix(self):
        return self._vector_to_R(self.direction)

    @property
    def tangent_1(self):
        n = self.direction
        if n[0] == 0 and n[1] == 0:
            return np.array([1, 0, 0])
        else:
            return self._normalize_vector(np.cross(n, np.array([0, 0, 1])))

    @property
    def tangent_2(self):
        return self._normalize_vector(np.cross(self.direction, self.tangent_1))

    def _RotAroundCenter(self, axis, theta):
        R = self.R(axis, theta)
        self.direction = np.dot(R, self.direction)
        return self

    def _RotAroundLocal(self, axis, localpoint, theta):
        R = self.R(axis, theta)
        localpoint = np.array(localpoint)
        self.direction = np.dot(R, self.direction)
        self.origin = self.origin + np.dot(R, -localpoint) + localpoint
        return self

    # >>> Gaussian Beam Functions
    def q_at_waist(self, w0):
        return GaussianBeam.q_at_waist(w0, self.wavelength, self.n)

    def q_at_z(self, z):
        return GaussianBeam.q_at_z(self.qo, z)

    def distance_to_waist(self, q):
        return GaussianBeam.distance_to_waist(q)

    def waist(self, q):
        return GaussianBeam.waist(q, self.wavelength, self.n)

    def rayleigh_range(self, q):
        return GaussianBeam.rayleigh_range(q)

    def radius_of_curvature(self, q):
        return GaussianBeam.radius_of_curvature(q)

    def spot_size(self, z):
        return GaussianBeam.spot_size(self.qo, z, self.wavelength, self.n)

    def Propagate(self, z):
        return self.copy(qo=self.q_at_z(z))

    def _render_sampling(self, length, num_points=10):
        z_to_waist = -self.distance_to_waist(self.qo)
        # waist is not in the ray path, linear sampling
        if z_to_waist < 0 or (z_to_waist > length):
            return np.linspace(0, length, num_points)
        # waist is in the ray path, non-linear sampling
        else:
            # sample at [-2,-1,0,1,2] of zR, adding start and end points
            # if some sampling points is not in [start, end], remove it
            t = z_to_waist + self.rayleigh_range(self.qo) * np.array([-2, -1, 0, 1, 2])
            t = list(t[(t >= 0) & (t <= length)])
            t.append(0)
            t.append(length)
            return np.sort(np.array(t))

    # <<< Gaussian Beam Functions

    def render(self, ax, type: str, **kwargs):
        """
        Render the ray in 2D.
        Each ray is represented by a line, and an arrow in the middle indicates direction.
        The alpha value reflects the intensity of the ray.

        Parameters:
            ax: Matplotlib 2D axis object for plotting.
            type: str, the type of rendering (e.g., "Z" for 2D, "3D" for 3D).
            kwargs: Additional arguments for customization (e.g., edge color).
        """

        # Set color and transparency based on ray properties
        # color = "blue" if self.length is None else "black"
        color = kwargs.get("color", "black")
        linewidth = kwargs.get("linewidth", 0.5)
        alpha = max(0.1, min(1.0, self.intensity))
        length = self.length if self.length else _RAY_NONE_LENGTH

        #
        if type == "Z":
            # Determine the start and end points of the ray
            start = self.origin[:2]
            end = start + self.direction[:2] * length

            # Plot the line representing the ray
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

            arrow = kwargs.get("ray_arrow", False)

            if arrow:
                # Add an arrow in the middle to indicate direction
                midpoint = (start + end) / 2
                arrow_length = 0.2 * np.linalg.norm(
                    end - start
                )  # Scale arrow relative to ray
                arrow_direction = (
                    arrow_length
                    * self.direction[:2]
                    / np.linalg.norm(self.direction[:2])
                )
                ax.arrow(
                    midpoint[0],
                    midpoint[1],
                    arrow_direction[0],
                    arrow_direction[1],
                    head_width=0.05 * np.linalg.norm(end - start),
                    head_length=0.1 * np.linalg.norm(end - start),
                    fc=color,
                    ec=color,
                    alpha=alpha,
                )
            #

            gaussian_beam = kwargs.get("gaussian_beam", False)
            if gaussian_beam:
                t = self._render_sampling(length, num_points=20)
                vx, vy = self.direction[0], self.direction[1]
                x = self.origin[0] + t * vx
                y = self.origin[1] + t * vy
                spot_size_scale = kwargs.get("spot_size_scale", 1.0)
                spots_size = self.spot_size(t) * spot_size_scale
                # create paths of polygon and fill it
                pts_x = np.concatenate(
                    [x + spots_size * (-vy), x[::-1] + spots_size[::-1] * (vy)]
                )
                pts_y = np.concatenate(
                    [y + spots_size * vx, y[::-1] + spots_size[::-1] * (-vx)]
                )
                ax.fill(pts_x, pts_y, color="red", alpha=alpha / 2, ec=None)
                #
                z_to_waist = -self.distance_to_waist(self.qo)
                if 0 < z_to_waist < length:
                    ax.scatter(
                        self.origin[0] + z_to_waist * vx,
                        self.origin[1] + z_to_waist * vy,
                        marker=".",
                        color="black",
                        alpha=alpha,
                    )

        elif type == "3D":
            # Determine the start and end points of the ray
            start = self.origin
            end = start + self.direction * length

            detailed_render = kwargs.get("detailed_render")
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )
            #
            arrow = kwargs.get("arrow", False)
            if arrow:
                # Add an arrow in the middle to indicate direction
                midpoint = (start + end) / 2
                arrow_length = 0.2 * np.linalg.norm(
                    end - start
                )  # Scale arrow relative to ray
                arrow_direction = arrow_length * self.direction
                ax.quiver(
                    midpoint[0],
                    midpoint[1],
                    midpoint[2],
                    arrow_direction[0],
                    arrow_direction[1],
                    arrow_direction[2],
                    pivot="middle",
                    color=color,
                    alpha=alpha,
                )

            gaussian_beam = kwargs.get("gaussian_beam", False)
            if gaussian_beam:
                vx, vy, vz = self.direction[0], self.direction[1], self.direction[2]
                z_to_waist = -self.distance_to_waist(self.qo)

                # Discrete z positions for the contour circles
                t = self._render_sampling(length, num_points=10)
                n_vertices = 6  # Number of vertices per circle (polygon approximation)

                # Compute the contour circles and store them in a list.
                if detailed_render:
                    circles = []
                    spot_size_scale = kwargs.get("spot_size_scale", 1.0)
                    spots_size = self.spot_size(t) * spot_size_scale
                    for t, w in zip(t, spots_size):
                        theta = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
                        center_pt = self.origin + t * self.direction
                        surface_pt = (
                            center_pt[
                                :, np.newaxis
                            ]  # Reshape center_pt to (3,1) for broadcasting
                            + w
                            * self.tangent_1[:, np.newaxis]
                            * np.cos(theta)  # Shape (3, n_vertices)
                            + w
                            * self.tangent_2[:, np.newaxis]
                            * np.sin(theta)  # Shape (3, n_vertices)
                        )
                        circle = surface_pt.T  # Transpose to get (n_vertices, 3) shape
                        # print(circle.shape)
                        # circle = np.column_stack((x, y, z))
                        circles.append(circle)

                    # Create side surfaces connecting adjacent circles.
                    faces = []
                    for i in range(len(circles) - 1):
                        c1 = circles[i]
                        c2 = circles[i + 1]
                        for j in range(n_vertices):
                            # Wrap around to the first vertex when j is the last index.
                            j_next = (j + 1) % n_vertices
                            # Define the quadrilateral face with 4 vertices:
                            face = [c1[j], c1[j_next], c2[j_next], c2[j]]
                            faces.append(face)

                    # Create a Poly3DCollection for the side surfaces.
                    side_collection = Poly3DCollection(
                        faces, facecolors="red", edgecolors=None, alpha=alpha / 2
                    )
                    ax.add_collection3d(side_collection)

                # draw a point at the waist position.`
                # print(self.origin, self.direction, z_to_waist)
                if 0 < z_to_waist < length:
                    ax.scatter(
                        self.origin[0] + z_to_waist * vx,
                        self.origin[1] + z_to_waist * vy,
                        self.origin[2] + z_to_waist * vz,
                        color="black",
                        alpha=alpha,
                    )
