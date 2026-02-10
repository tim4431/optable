from .base import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .material import *

_RAY_NONE_LENGTH = 100


class GaussianBeam:
    """Utility formulas for Gaussian beam ``q``-parameter optics."""

    @staticmethod
    def q_at_waist(w0, wl, n=1):
        """Return complex ``q`` at beam waist from waist radius and wavelength."""
        return (1j * n * np.pi * w0**2) / wl

    @staticmethod
    def q_at_z(qo, z):
        """Propagate ``q`` by free-space distance ``z``."""
        return qo + z

    @staticmethod
    def distance_to_waist(q):
        """Return signed distance from current plane to waist plane."""
        z = np.real(q)
        return z

    @staticmethod
    def waist(q, wl, n=1):
        """Return waist radius implied by complex ``q``."""
        w0 = np.sqrt((wl * np.imag(q)) / (n * np.pi))
        return float(w0)

    @staticmethod
    def rayleigh_range(q):
        """Return Rayleigh range from complex ``q``."""
        zr = np.imag(q)
        return float(zr)

    @staticmethod
    def radius_of_curvature(q):
        """Return wavefront radius of curvature from complex ``q``."""
        R = 1 / np.real(1 / q)
        return float(R)

    @staticmethod
    def spot_size(qo, z, wl, n=1):
        """Return beam spot size after propagation distance ``z``."""
        q = qo + z
        w = np.sqrt(-wl / (n * np.pi * np.imag(1 / q)))
        return w


class Ray(Vector):
    """Geometric ray with optional Gaussian beam state."""

    _n = RefractiveIndex("_n")  # refractive index

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
        """Initialize a ray.

        Args:
            origin: Ray origin in lab coordinates.
            direction: Propagation direction (normalized internally).
            intensity: Relative ray intensity.
            wavelength: Wavelength in model units.
            length: Optional finite rendering/segment length.
            alive: If ``False``, ray is ignored in interaction steps.
            qo: Optional Gaussian beam complex ``q`` at origin.
            w0: Optional beam waist radius used to derive ``qo``.
        """
        super().__init__(origin, **kwargs)
        self.length = float(length) if length else None  # Length of the ray
        self.direction = direction  # Direction vector
        self.intensity = float(intensity)  # Intensity of the ray
        self.wavelength = (
            float(wavelength) if wavelength else 0.0
        )  # Wavelength of the ray
        self.alive = alive  # Ray is alive means it has not been absorbed or terminated
        self._n = 1.0  # default vacuum
        self._pathlength = 0.0  # total path length traveled by the ray from its start, at t=0, count in vacuum
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
        """Unit propagation direction vector."""
        return self._direction

    @direction.setter
    def direction(self, direction):
        """Set direction from an arbitrary vector and normalize."""
        self._direction = self._normalize_vector(direction)
        # print(self._direction)

    @property
    def n(self):
        """Return refractive index at current wavelength."""
        return self._n()

    @property
    def transform_matrix(self):
        """Return rotation matrix mapping local ray frame to lab frame."""
        return self._vector_to_R(self.direction)

    @property
    def tangent_1(self):
        """Return first unit tangent vector orthogonal to propagation."""
        n = self.direction
        if n[0] == 0 and n[1] == 0:
            return np.array([1, 0, 0])
        else:
            return self._normalize_vector(np.cross(n, np.array([0, 0, 1])))

    @property
    def tangent_2(self):
        """Return second unit tangent vector orthogonal to propagation."""
        return self._normalize_vector(np.cross(self.direction, self.tangent_1))

    def pathlength(self, t: float = 0) -> float:
        """Return accumulated optical path length including offset ``t``."""
        return float(self._pathlength + t * self.n)

    def phase(self, t: float = 0) -> float:
        """Return optical phase modulo ``2*pi`` at offset ``t``."""
        return np.mod((2 * np.pi / self.wavelength) * self.pathlength(t), 2 * np.pi)

    def _RotAroundLocal(self, axis, localpoint, theta):
        R = self.R(axis, theta)
        localpoint = np.array(localpoint)
        self.direction = np.dot(R, self.direction)
        self.origin = self.origin + np.dot(R, -localpoint) + localpoint
        return self

    # >>> Gaussian Beam Functions
    def q_at_waist(self, w0):
        """Return Gaussian ``q`` at waist radius ``w0`` for this ray."""
        return GaussianBeam.q_at_waist(w0, self.wavelength, self.n)

    def q_at_z(self, z):
        """Return Gaussian ``q`` after propagation distance ``z``."""
        return GaussianBeam.q_at_z(self.qo, z)

    def distance_to_waist(self, q):
        """Return distance from given ``q`` plane to waist plane."""
        return GaussianBeam.distance_to_waist(q)

    def waist(self, q):
        """Return waist radius implied by ``q`` for this wavelength/index."""
        return GaussianBeam.waist(q, self.wavelength, self.n)

    def rayleigh_range(self, q):
        """Return Rayleigh range from ``q``."""
        return GaussianBeam.rayleigh_range(q)

    def radius_of_curvature(self, q):
        """Return wavefront curvature radius from ``q``."""
        return GaussianBeam.radius_of_curvature(q)

    def spot_size(self, z):
        """Return spot size at propagation distance ``z``."""
        return GaussianBeam.spot_size(self.qo, z, self.wavelength, self.n)

    def Propagate(self, z):
        """Return a copied ray with ``qo`` propagated by distance ``z``."""
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
        """Render the ray in 2D (``"Z"``) or 3D (``"3D"``).

        Supports optional directional arrows and Gaussian beam envelopes.
        """

        # Set color and transparency based on ray properties
        # color = "blue" if self.length is None else "black"
        physical_color = kwargs.get("physical_color", False)
        if physical_color and self.wavelength is not None:
            color = wavelength_to_rgb(self.wavelength * self.unit)
        else:
            color = kwargs.get("color", "black")
        linewidth = kwargs.get("linewidth", 0.5)
        linestyle = kwargs.get("linestyle", "-")
        alpha = max(0.1, min(1.0, self.intensity))
        length = self.length if self.length else _RAY_NONE_LENGTH

        #
        if type == "Z":
            # Determine the start and end points of the ray
            start = self.origin[:2]
            end = start + self.direction[:2] * length

            render_line = kwargs.get("render_line", True)
            # Plot the line representing the ray
            if render_line:
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    linestyle=linestyle,
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
                    linestyle=linestyle,
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
                fill_color = "red" if not physical_color else color
                ax.fill(pts_x, pts_y, color=fill_color, alpha=alpha / 2, ec=None)
                #
                z_to_waist = -self.distance_to_waist(self.qo)
                annote_waist = kwargs.get("annote_waist", True)
                if annote_waist:
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
                    linestyle=linestyle,
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


def multiplex_rays_in_wavelength(
    rays: List[Ray], wavelength_list: List[float]
) -> List[Ray]:
    """Create multiple rays with different wavelengths from a single ray.

    Args:
        rays (List[Ray]): List of Ray objects to be multiplexed.
        wavelength_list (List[float]): List of wavelengths to assign to the rays.

    Returns:
        List[Ray]: New list of Ray objects with assigned wavelengths.
    """
    multiplexed_rays = []
    for wl in wavelength_list:
        for ray in rays:
            new_ray = ray.copy(wavelength=wl)
            multiplexed_rays.append(new_ray)
    return multiplexed_rays
