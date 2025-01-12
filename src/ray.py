from base import *

RAY_NONE_LENGTH = 60


class Ray(Vector):
    def __init__(
        self,
        origin,
        direction,
        intensity: float = 1.0,
        wavelength=None,
        length=None,
        alive=True,
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

    def __repr__(self):
        return f"Ray(origin={self.origin}, direction={self.direction}, intensity={self.intensity}, length={self.length}, alive={self.alive})"

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        self._direction = self._normalize_vector(direction)

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

    def render(self, ax, type: str, **kwargs):
        """
        Render the ray in 2D.
        Each ray is represented by a line, and an arrow in the middle indicates direction.
        The alpha value reflects the intensity of the ray.

        Parameters:
            ax: Matplotlib 2D axis object for plotting.
        """

        # Set color and transparency based on ray properties
        # color = "blue" if self.length is None else "black"
        color = "black"
        alpha = max(0.1, min(1.0, self.intensity))
        linewidth = kwargs.get("linewidth", 0.5)
        #
        #
        if type == "Z":
            # Determine the start and end points of the ray
            start = self.origin[:2]
            if self.length is not None:
                end = start + self.length * self.direction[:2]
            else:
                end = start + self.direction[:2] * RAY_NONE_LENGTH

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

        elif type == "3D":
            # Determine the start and end points of the ray
            start = self.origin
            if self.length is not None:
                end = start + self.length * self.direction
            else:
                end = start + self.direction * RAY_NONE_LENGTH

            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )
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


class GaussianBeam(Ray):
    def __init__(
        self,
        origin,
        direction,
        intensity: float = 1.0,
        wavelength=None,
        qo=None,  # q at origin
        w0=None,  # if specified, qo will be calculated from w0
        n=1.0,
        **kwargs,
    ):
        super().__init__(
            origin,
            direction,
            intensity,
            wavelength=wavelength,
            **kwargs,
        )
        self.wavelength = wavelength
        self.n = n  # refractive index
        if qo is not None:
            self.qo = qo  # q at origin
        else:
            self.qo = self.q_at_waist(w0)

    def q_at_waist(self, w0):
        return (1j * np.pi * w0**2) / self.wavelength

    def q_at_z(self, z):
        return self.qo + z

    def distance_to_waist(self, q):
        z = np.real(q)
        return float(z)

    def waist(self, q):
        w0 = np.sqrt((self.wavelength * np.imag(q)) / (self.n * np.pi))
        return float(w0)

    def rayleigh_range(self, q):
        zr = np.imag(q)
        return float(zr)

    def radius_of_curvature(self, q):
        R = 1 / np.real(1 / q)
        return float(R)

    def spot_size(self, z):
        q = self.q_at_z(z)
        w = np.sqrt(-self.wavelength / (self.n * np.pi * np.imag(1 / q)))
        return w

    def Propagate(self, z):
        return self.copy(qo=self.q_at_z(z))
