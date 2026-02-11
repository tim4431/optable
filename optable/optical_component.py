import scipy
from .base import *
from .ray import *
from .surfaces import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class OpticalComponent(Vector):
    """Base class for optical elements represented in a local coordinate frame."""

    def __init__(self, origin, **kwargs):
        """Initialize a component at a lab-frame origin.

        Args:
            origin: Component origin in lab coordinates.
            **kwargs: Optional display metadata such as ``name``, ``label``,
                ``render_obj``, and ``render_comp_vec``.
        """
        super().__init__(origin, **kwargs)
        self.transform_matrix = np.identity(3)
        self.surface = Plane()  # Default surface is a plane
        self._bbox = (
            None,
            None,
            None,
            None,
            None,
            None,
        )  # xmin, xmax, ymin, ymax, zmin, zmax
        self.render_obj = kwargs.get("render_obj", True)
        self.render_comp_vec = kwargs.get("render_comp_vec", False)
        self.name = kwargs.get("name", None)
        self.label = kwargs.get("label", None)
        self.label_position = kwargs.get(
            "label_position", [1, 0, 0]
        )  # [dx,dy] or [dx,dy,dz]
        self._interact_count = {}
        self.max_interact_count = kwargs.get("max_interact_count", None)

    def __repr__(self):
        return f"OpticalComponent(origin={self.origin}, transform_matrix=\n{self.transform_matrix})"

    # normal in local fram is always x-axis
    @property
    def normal(self):
        """Return the surface normal in lab coordinates."""
        return self.transform_matrix @ np.array([1, 0, 0])

    @property
    def tangent_Y(self):
        """Return local +Y axis expressed in lab coordinates."""
        return self.transform_matrix @ np.array([0, 1, 0])

    @property
    def tangent_Z(self):
        """Return local +Z axis expressed in lab coordinates."""
        return self.transform_matrix @ np.array([0, 0, 1])

    def get_bbox_local(self):
        raise NotImplementedError("Subclasses must implement get_bbox_local method")

    @property
    def bbox(self):
        """Return cached axis-aligned bounding box in lab coordinates."""
        if self._bbox == (None, None, None, None, None, None):
            self._bbox = tuple(self.get_bbox())
        return tuple(self._bbox)

    def get_bbox(self) -> tuple:
        """Compute axis-aligned bounding box in lab frame.

        Returns:
            Tuple ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in lab coordinates.
        """
        bbox_local = self.get_bbox_local()
        corners_local = np.array(
            [
                [bbox_local[0], bbox_local[2], bbox_local[4]],  # [xmin, ymin, zmin]
                [bbox_local[1], bbox_local[2], bbox_local[4]],  # [xmax, ymin, zmin]
                [bbox_local[0], bbox_local[3], bbox_local[4]],  # [xmin, ymax, zmin]
                [bbox_local[1], bbox_local[3], bbox_local[4]],  # [xmax, ymax, zmin]
                [bbox_local[0], bbox_local[2], bbox_local[5]],  # [xmin, ymin, zmax]
                [bbox_local[1], bbox_local[2], bbox_local[5]],  # [xmax, ymin, zmax]
                [bbox_local[0], bbox_local[3], bbox_local[5]],  # [xmin, ymax, zmax]
                [bbox_local[1], bbox_local[3], bbox_local[5]],  # [xmax, ymax, zmax]
            ]
        ).T  # shape (3, 8)
        corners_global = (self.transform_matrix @ corners_local) + self.origin.reshape(
            -1, 1
        )  # shape (3, 8)
        xmin, xmax = np.min(corners_global[0, :]), np.max(corners_global[0, :])
        ymin, ymax = np.min(corners_global[1, :]), np.max(corners_global[1, :])
        zmin, zmax = np.min(corners_global[2, :]), np.max(corners_global[2, :])
        # print(
        #     f"Component {self.name}, bbox: {(xmin, xmax, ymin, ymax, zmin, zmax)},origin: {self.origin}"
        # )
        return (xmin, xmax, ymin, ymax, zmin, zmax)

    def _RotAroundLocal(self, axis, localpoint, theta):
        R = self.R(axis, theta)
        localpoint = np.array(localpoint)
        self.transform_matrix = np.dot(R, self.transform_matrix)
        self.origin = self.origin + np.dot(R, -localpoint) + localpoint
        return self

    def ray_to_local_coordinates(self, ray: Ray) -> Ray:
        """Transform a ray from lab frame into this component's local frame."""
        R = np.linalg.inv(self.transform_matrix)
        local_origin = np.dot(R, ray.origin - self.origin)
        local_direction = np.dot(R, ray.direction)
        return ray.copy(origin=local_origin, direction=local_direction)

    def point_to_lab_coordinates(self, point_local: np.ndarray) -> np.ndarray:
        """Transform a point from local frame back to the lab frame."""
        R = self.transform_matrix
        global_point = np.dot(R, point_local) + self.origin
        return global_point

    def ray_to_lab_coordinates(self, ray: Ray) -> Ray:
        """Transform a ray from local frame back to the lab frame."""
        R = self.transform_matrix
        global_origin = np.dot(R, ray.origin) + self.origin
        global_direction = np.dot(R, ray.direction)
        return ray.copy(origin=global_origin, direction=global_direction)

    def _find_zero(self, f, a, b, num_intervals: int = 10) -> np.ndarray:
        t_values = np.linspace(a, b, int(num_intervals))
        sols = []
        for i in range(len(t_values) - 1):
            t_start, t_end = t_values[i], t_values[i + 1]
            if f(t_start) * f(t_end) < 0:
                t = scipy.optimize.brentq(f, t_start, t_end)
                sols.append(t)
        return np.array(sols)

    def get_interact_count(self, ray_id):
        count = self._interact_count.get(ray_id, 0)
        return count

    def should_interact(self, ray_id):
        if self.max_interact_count is None:
            return True
        else:
            count = self._interact_count.get(ray_id, 0)
            return count < self.max_interact_count

    def increase_interact_count(self, ray_id):
        count = self._interact_count.get(ray_id, 0)
        self._interact_count[ray_id] = count + 1

    def intersect_point_local(
        self, ray: Ray
    ) -> Union[Tuple[np.ndarray, float], Tuple[None, None]]:
        """Find first valid intersection with the component in local coordinates.

        Args:
            ray: Input ray already expressed in the component local frame.

        Returns:
            ``(P, t)`` where ``P`` is the local intersection point and ``t`` is
            the ray parameter. Returns ``(None, None)`` if no valid hit exists.
        """
        EPS = 1e-9
        #
        if self.surface.planar:
            # rays intesection with the plane, solve the equation
            # (P - P0) . n = 0, where P is the intersection point, P0 is the origin of optical component in local frame (0, 0, 0)
            # P = Ps + t * v, where v is the direction of the ray, Ps is the origin of the ray

            # t = (P0 - Ps) . n / v . n
            numerator = np.dot(-ray.origin, np.array([1, 0, 0]))
            denominator = np.dot(ray.direction, np.array([1, 0, 0]))
            if denominator == 0:
                if numerator != 0:
                    return None, None
                else:
                    t = 0
            else:
                t = numerator / denominator
            #
            # Case 0: t=0, the just outgoing ray interact with the optical component
            # Case 1: t>ray.length, has been absorbed
            # Case 2: t<0, backward ray
            if (
                (t is None)
                or (np.abs(t) < EPS)
                or (t < 0)
                or ((ray.length is not None) and (t > ray.length))
            ):
                return None, None
            else:
                P = ray.origin + t * ray.direction
                if not (self.surface.within_boundary(P)):
                    return None, None
                else:
                    return P, t
        else:
            # P = (x, y, z)
            # f(P) = 0
            # P = Ps + t * v
            # -> f(Ps + t * v) = 0, numerically solve t
            def f(t):
                Pt = ray.origin + t * ray.direction
                return self.surface.f(Pt)

            t1, t2, hits = self.surface.solve_crosssection_ray_bbox_local(
                ray.origin, ray.direction
            )
            t1, t2 = float(t1[0]), float(t2[0])
            # print("t1, t2", t1, t2)
            if t2 + EPS < t1:
                return None, None
            t1 = max(t1, 0)
            t2 = min(t2, 100)
            tList = self._find_zero(f, t1 - EPS, t2 + EPS)

            #
            if len(tList) == 0:
                return None, None
            else:
                valid_mask_0 = (tList >= 0) & (np.abs(tList) >= EPS)
                if ray.length is not None:
                    valid_mask_1 = tList <= ray.length
                    valid_mask = valid_mask_0 & valid_mask_1
                else:
                    valid_mask = valid_mask_0
                tvalid = tList[valid_mask]
                #
                for t in np.sort(tvalid):
                    P = ray.origin + t * ray.direction
                    if self.surface.within_boundary(P):
                        return P, t
                return None, None

    def interact_local(self, ray: Ray) -> List[Ray]:
        """Compute outgoing rays in local coordinates.

        Subclasses must implement optical behavior using local-frame geometry.
        """
        raise NotImplementedError("Subclasses must implement interact_local method")

    def _get_boundary_points(self, type: str):
        t = np.linspace(0, 1, 100)
        points_local = self.surface.parametric_boundary(t, type)
        points_global = (self.transform_matrix @ points_local) + self.origin.reshape(
            -1, 1
        )
        # print(points_global)
        return points_global

    def render(self, ax, type: str, **kwargs):
        """Render the component boundary in 2D (``"Z"``) or 3D (``"3D"``)."""
        if not self.render_obj:
            return
        # Get edge color and line width from kwargs
        color = kwargs.get("color", "black")
        linewidth = kwargs.get("linewidth", 2)
        global_x, global_y, global_z = self._get_boundary_points(type)
        detailed_render = kwargs.get("detailed_render", False)
        label = kwargs.get("label", None)
        label_fontsize = kwargs.get("label_fontsize", 10)
        #
        if type == "Z":
            ax.plot(
                global_x,
                global_y,
                color=color,
                linewidth=linewidth,
            )
            if self.render_comp_vec:
                # add a component vector (red)
                normal = self.normal
                ax.quiver(
                    self.origin[0],
                    self.origin[1],
                    normal[0],
                    normal[1],
                    color=color,
                    # scale=2,
                    # scale_units="xy",
                )
            if label and self.label:
                if self.label_position is not None:
                    ax.text(
                        self.origin[0] + self.label_position[0],
                        self.origin[1] + self.label_position[1],
                        self.label,
                        color=color,
                        fontsize=label_fontsize,
                    )

        elif type == "3D":
            ax.plot(
                global_x,
                global_y,
                global_z,
                color=color,
                linewidth=linewidth,
            )
            if self.render_comp_vec:
                # add a component vector (red)
                normal = self.normal
                ax.quiver(
                    self.origin[0],
                    self.origin[1],
                    self.origin[2],
                    normal[0],
                    normal[1],
                    normal[2],
                    color=color,
                    # scale=2,
                    # scale_units="xy",
                )
            if detailed_render:
                # add a 3D polygon
                poly = Poly3DCollection(
                    [list(zip(global_x, global_y, global_z))],
                    color=color,
                    linewidths=linewidth,
                    edgecolors=color,
                    alpha=0.25,
                )
                ax.add_collection3d(poly)

        else:
            raise ValueError(f"render: Invalid type: {type}")

    def interact(self, ray: Ray) -> Union[Tuple[float, List[Ray]], Tuple[None, None]]:
        """Apply component interaction in local frame and return lab-frame rays.

        Args:
            ray: Input ray in lab coordinates.

        Returns:
            Tuple ``(t, rays)`` where ``t`` is the hit distance and ``rays``
            contains the truncated incoming segment plus newly generated outgoing
            rays in lab coordinates. Returns ``(None, None)`` when no interaction
            occurs.
        """
        if not ray.alive:
            return None, None
        else:
            local_ray = self.ray_to_local_coordinates(ray)
            P, t = self.intersect_point_local(local_ray)
            #

            if P is None:
                return None, None
            else:
                if self.should_interact(ray._id):
                    self.increase_interact_count(ray._id)
                else:
                    return None, None
                #
                truncated_ray = ray.copy(length=t, alive=False)
                # new rays after interaction
                local_rays_after_interaction = self.interact_local(
                    local_ray
                )  # List[Ray]
                lab_rays_after_interaction = [
                    self.ray_to_lab_coordinates(local_ray)
                    for local_ray in local_rays_after_interaction
                ]
                # global_point = self.point_to_lab_coordinates(P)

                # print(
                #     f"Component {self.name} interacted with ray {ray._id}, intensity {ray.intensity} at global point {global_point} and distance {t}, generated {len(lab_rays_after_interaction)} rays"
                # )
                return t, [truncated_ray] + lab_rays_after_interaction

    def patch_block(self, width, height):
        """Create a ``Block`` patch sharing this component pose and aperture."""
        obj = Block(self.origin, hole=self.surface, width=width, height=height)
        obj.transform_matrix = self.transform_matrix
        return obj

    def gather_components(
        self, avoid_flatten_classname: List = [], ignore_classname: List = []
    ) -> List:
        """Return a flattened metadata list for this component hierarchy.

        Args:
            avoid_flatten_classname: Class names whose children are not expanded.
            ignore_classname: Class names excluded from output.

        Returns:
            A list of serializable dictionaries describing components.
        """

        def _repr_self_dict():
            d = {
                "name": get_attr_str(self, "name", "None"),
                "class": self.__class__.__name__,
                "origin": to_mathematical_str(str(self.origin.tolist())),
                "transform_matrix": to_mathematical_str(
                    str(self.transform_matrix.tolist())
                ),
                "radius": get_attr_str(self, "radius", "None"),
                "width": get_attr_str(self, "width", "None"),
                "height": get_attr_str(self, "height", "None"),
                "focal_length": get_attr_str(self, "focal_length", "None"),
            }
            return d

        components = []
        if self.__class__.__name__ not in ignore_classname:
            components.append(_repr_self_dict())
        if hasattr(self, "components"):
            for component in self.components:
                if self.__class__.__name__ not in avoid_flatten_classname:
                    components.extend(
                        component.gather_components(
                            avoid_flatten_classname=avoid_flatten_classname,
                            ignore_classname=ignore_classname,
                        )
                    )
        return components


class PointObj(OpticalComponent):
    """Degenerate point-like marker component that transmits all rays."""

    def __init__(self, origin, **kwargs):
        """Initialize a point object at ``origin``."""
        super().__init__(origin, **kwargs)
        self.surface = Point()
        self._edge_color = "orange"

    def interact_local(self, ray):
        """Return the same ray to model perfect transmission."""
        return [ray]  # every ray is transmitted

    def render(self, ax, type: str, **kwargs):
        """Render the point marker in 2D or 3D."""
        if type == "Z":
            ax.scatter(
                self.origin[0], self.origin[1], color=self._edge_color, marker="+", s=20
            )
        elif type == "3D":
            ax.scatter(
                self.origin[0],
                self.origin[1],
                self.origin[2],
                color=self._edge_color,
                marker="+",
                s=20,
            )

    def get_bbox_local(self):
        """Return local bounding box from point surface geometry."""
        return self.surface.get_bbox_local()


class Block(OpticalComponent):
    """Opaque rectangular blocker with an optional hole aperture."""

    def __init__(
        self,
        origin,
        hole: Union[Surface | None] = None,
        width: float = 1.0,
        height: float = 1.0,
        **kwargs,
    ):
        """Initialize a blocking rectangle.

        Args:
            origin: Component origin in lab coordinates.
            hole: Optional surface removed from the rectangle aperture.
            width: Rectangle width along local Y.
            height: Rectangle height along local Z.
        """
        super().__init__(origin, **kwargs)
        self.width = width
        self.height = height
        self.surface = (
            Rectangle(width, height).subtract(hole)
            if hole is not None
            else Rectangle(width, height)
        )
        self._edge_color = "black"

    def interact_local(self, ray):
        """Absorb all incident rays."""
        return []  # every ray is absorbed

    def render(self, ax, type: str, **kwargs):
        """Render block outline."""
        super().render(ax, type, color=self._edge_color, **kwargs)

    def get_bbox_local(self):
        """Return local bounding box from rectangle geometry."""
        return self.surface.get_bbox_local()


class BaseMirror(OpticalComponent):
    """Base reflective surface with optional straight-through transmission."""

    def __init__(
        self,
        origin,
        reflectivity: float = 1.0,
        transmission: float = 0.0,
        **kwargs,
    ):
        """Initialize mirror interaction coefficients.

        Args:
            origin: Component origin in lab coordinates.
            reflectivity: Fraction of incoming intensity sent to reflected ray.
            transmission: Fraction of incoming intensity sent to transmitted ray.
        """
        super().__init__(origin, **kwargs)
        self.reflectivity = reflectivity
        self.transmission = transmission
        #
        self._edge_color = "green"

    def interact_local(self, ray):
        """Compute reflected/transmitted rays in local coordinates.

        The input ``ray`` must already be in this component local frame. A
        reflected branch is generated when ``reflectivity > 0`` and a
        straight-through branch when ``transmission > 0``.
        """
        P, t = self.intersect_point_local(ray)
        normal = self.surface.normal(P)
        #
        rays = []
        qo = None if ray.qo is None else ray.q_at_z(t)
        if self.reflectivity > 0:
            reflected_direction = (
                ray.direction - 2 * np.dot(ray.direction, normal) * normal
            )
            reflected_ray = ray.copy(
                origin=P,
                direction=reflected_direction,
                intensity=ray.intensity * self.reflectivity,
                qo=qo,
                _pathlength=ray.pathlength(float(t)),
            )
            rays.append(reflected_ray)
        if self.transmission > 0:
            transmitted_ray = ray.copy(
                origin=P,
                direction=ray.direction,
                intensity=ray.intensity * self.transmission,
                qo=qo,
                _pathlength=ray.pathlength(float(t)),
            )
            rays.append(transmitted_ray)
        #
        return rays

    def render(self, ax, type: str, **kwargs):
        """Render mirror outline."""
        super().render(ax, type, color=self._edge_color, **kwargs)

    def get_bbox_local(self):
        """Return local bounding box from the active surface."""
        return self.surface.get_bbox_local()


class BaseRefraciveSurface(OpticalComponent):
    """Base interface between two refractive media."""

    _n1 = RefractiveIndex("_n1")
    _n2 = RefractiveIndex("_n2")

    def __init__(
        self,
        origin,
        n1: Union[float, Material] = 1.0,
        n2: Union[float, Material] = 1.0,
        reflectivity: float = 0.0,
        transmission: float = 1.0,
        **kwargs,
    ):
        """Initialize refractive interface parameters.

        Args:
            origin: Component origin in lab coordinates.
            n1: Refractive index for local ``x > 0`` side.
            n2: Refractive index for local ``x < 0`` side.
            reflectivity: Additional reflected branch coefficient.
            transmission: Transmitted branch coefficient.
            **kwargs: Optional ``surface`` override and rendering options.
        """
        super().__init__(origin, **kwargs)

        self._n1 = n1
        self._n2 = n2
        #
        self.reflectivity = reflectivity
        self.transmission = transmission
        #
        self._edge_color = "gray"
        self.surface = kwargs.get("surface", Plane())
        self.roc = self.surface.roc if hasattr(self.surface, "roc") else np.inf

    def interact_local(self, ray):
        """Apply Snell refraction and optional reflection in local frame.

        The method determines incident side from the local surface normal,
        computes transmitted direction using Snell's law, handles total internal
        reflection, and updates Gaussian beam ``q`` with ABCD matrices when
        available.
        """
        P, t = self.intersect_point_local(ray)
        normal = self.surface.normal(P)
        n1 = self._n1(ray.wavelength * ray.unit)
        n2 = self._n2(ray.wavelength * ray.unit)
        #
        rays = []
        #
        # ROC is positive if center of curvature is toward nout side
        ROC = np.inf
        if hasattr(self, "roc"):
            # if the roc is a callable function
            if callable(self.roc):
                ROC = self.roc(P)
            else:
                ROC = self.roc

        # print("ROC:", ROC)
        if np.dot(ray.direction, normal) < 0:  # incident from n1 to n2
            nin, nout = n1, n2
        else:  # incident from n2 to n1
            nin, nout = n2, n1
            ROC = -ROC  # change the sign of ROC
        #
        ABCD_refraction = np.array([[1, 0], [(nin - nout) / (ROC * nout), nin / nout]])
        ABCD_reflection = np.array([[1, 0], [2 / ROC, 1]])

        #
        if ray.qo is not None:
            qin = ray.q_at_z(t)
            qo_trans = (ABCD_refraction[0, 0] * qin + ABCD_refraction[0, 1]) / (
                ABCD_refraction[1, 0] * qin + ABCD_refraction[1, 1]
            )
        else:
            qo_trans = None
        #
        if ray.qo is not None:
            qin = ray.q_at_z(t)
            qo_refl = (ABCD_reflection[0, 0] * qin + ABCD_reflection[0, 1]) / (
                ABCD_reflection[1, 0] * qin + ABCD_reflection[1, 1]
            )
        else:
            qo_refl = None

        r_n = np.dot(ray.direction, normal)
        r_t = ray.direction - r_n * normal
        if r_n > 0:
            transmitted_normal = normal
        else:
            transmitted_normal = -normal
        cos_theta_i = r_n
        cos_theta_i = np.clip(cos_theta_i, -1, 1)  # avoid numerical issues
        sin_theta_i = np.sqrt(1 - cos_theta_i**2)
        sin_theta_t = (nin * sin_theta_i) / nout
        if sin_theta_t < 1:
            if self.transmission > 0:
                cos_theta_t = np.sqrt(1 - sin_theta_t**2)
                transmitted_direction = (
                    nin / nout
                ) * r_t + cos_theta_t * transmitted_normal
                transmitted_ray = ray.copy(
                    origin=P,
                    direction=transmitted_direction,
                    intensity=ray.intensity * self.transmission,
                    qo=qo_trans,
                    _n=nout,
                    _pathlength=ray.pathlength(float(t)),
                )
                rays.append(transmitted_ray)
        else:
            # total internal reflection
            reflected_direction = ray.direction + 2 * cos_theta_i * (-normal)
            reflected_ray = ray.copy(
                origin=P,
                direction=reflected_direction,
                intensity=ray.intensity,
                qo=qo_refl,
                _pathlength=ray.pathlength(float(t)),
            )
            rays.append(reflected_ray)

        #
        if self.reflectivity > 0:
            reflected_direction = ray.direction + 2 * cos_theta_i * (-normal)
            reflected_ray = ray.copy(
                origin=P,
                direction=reflected_direction,
                intensity=ray.intensity * self.reflectivity,
                qo=qo_refl,
                _pathlength=ray.pathlength(float(t)),
            )
            rays.append(reflected_ray)
        #
        return rays

    def render(self, ax, type: str, **kwargs):
        """Render refractive surface outline."""
        super().render(ax, type, color=self._edge_color, **kwargs)

    def get_bbox_local(self):
        """Return local bounding box from the active surface."""
        return self.surface.get_bbox_local()


class Mirror(BaseMirror):
    """Circular mirror."""

    def __init__(
        self,
        origin,
        radius: float = 0.5,
        reflectivity: float = 1.0,
        transmission: float = 0.0,
        **kwargs,
    ):
        """Initialize a circular mirror."""
        super().__init__(
            origin, reflectivity=reflectivity, transmission=transmission, **kwargs
        )
        self.radius = radius
        self.surface = Circle(radius)


class SquareMirror(BaseMirror):
    """Rectangular mirror."""

    def __init__(
        self,
        origin,
        width: float = 1.0,
        height: float = 1.0,
        reflectivity: float = 1.0,
        transmission: float = 0.0,
        **kwargs,
    ):
        """Initialize a rectangular mirror."""
        super().__init__(
            origin, reflectivity=reflectivity, transmission=transmission, **kwargs
        )
        self.width = width
        self.height = height
        self.surface = Rectangle(width, height)


class SquareRefractive(BaseRefraciveSurface):
    """Rectangular refractive interface."""

    def __init__(
        self,
        origin,
        width: float = 1.0,
        height: float = 1.0,
        n1: Union[float, Material] = 1.0,
        n2: Union[float, Material] = 1.0,
        reflectivity: float = 0.0,
        transmission: float = 1.0,
        **kwargs,
    ):
        """Initialize a rectangular refractive surface."""
        super().__init__(
            origin,
            n1=n1,
            n2=n2,
            reflectivity=reflectivity,
            transmission=transmission,
            **kwargs,
        )
        self.width = width
        self.height = height
        self.surface = Rectangle(width, height)


class CircleRefractive(BaseRefraciveSurface):
    """Circular refractive interface."""

    def __init__(
        self,
        origin,
        radius: float = 0.5,
        n1: Union[float, Material] = 1.0,
        n2: Union[float, Material] = 1.0,
        reflectivity: float = 0.0,
        transmission: float = 1.0,
        **kwargs,
    ):
        """Initialize a circular refractive surface."""
        super().__init__(
            origin,
            n1=n1,
            n2=n2,
            reflectivity=reflectivity,
            transmission=transmission,
            **kwargs,
        )
        self.radius = radius
        self.surface = Circle(radius)


class SphereRefractive(BaseRefraciveSurface):
    """Spherical-cap refractive interface."""

    def __init__(
        self,
        origin,
        radius: float = 0.5,
        height: float = 0.5,
        n1: Union[float, Material] = 1.0,
        n2: Union[float, Material] = 1.0,
        reflectivity: float = 0.0,
        transmission: float = 1.0,
        **kwargs,
    ):
        """Initialize a spherical refractive surface."""
        super().__init__(
            origin,
            n1=n1,
            n2=n2,
            reflectivity=reflectivity,
            transmission=transmission,
            **kwargs,
        )
        self.radius = radius
        self.height = height
        self.roc = radius
        self.surface = Sphere(radius, height)


class BeamSplitter(SquareMirror):
    """Rectangular beamsplitter modeled as a partially reflective mirror."""

    def __init__(self, origin, width=1.0, height=1.0, eta: float = 0.5, **kwargs):
        """Initialize beamsplitter from splitting ratio ``eta``.

        Args:
            origin: Component origin in lab coordinates.
            width: Aperture width.
            height: Aperture height.
            eta: Power ratio sent to reflected branch.
        """
        super().__init__(
            origin,
            width=width,
            height=height,
            reflectivity=np.sqrt(eta),
            transmission=np.sqrt(1 - eta),
            **kwargs,
        )
        edgecolor = kwargs.get("edgecolor", Color.SCIENCE_BLUE_DARK)
        self._edge_color = edgecolor

    def render(self, ax, type, **kwargs):
        """Render the beamsplitter boundary and optional filled face in 2D."""
        super().render(ax, type, **kwargs)
        facecolor = kwargs.get("facecolor", Color.SCIENCE_BLUE_LIGHT)
        linewidth = kwargs.get("linewidth", 2)
        # draw the outer frame with edgecolor, inner cube with facecolor
        if type == "Z":
            rect_pts = [
                [-self.width / 2, 0, 0],
                [0, -self.width / 2, 0],
                [self.width / 2, 0, 0],
                [0, self.width / 2, 0],
            ]
            # print(np.array(rect_pts).shape)
            rect_pts = self.transform_matrix @ np.transpose(
                np.array(rect_pts)
            ) + np.array(self.origin).reshape(-1, 1)
            # print(rect_pts.shape)
            ax.add_patch(
                plt.Polygon(
                    np.transpose(rect_pts)[:, :2],
                    facecolor=facecolor,
                    edgecolor=self._edge_color,
                    linewidth=linewidth,
                )
            )


class Lens(OpticalComponent):
    """Thin lens with circular aperture."""

    def __init__(
        self,
        origin,
        focal_length,
        radius: float = 0.5,
        transmission: float = 1.0,
        **kwargs,
    ):
        """Initialize a thin lens element.

        Args:
            origin: Component origin in lab coordinates.
            focal_length: Thin-lens focal length in model units.
            radius: Circular aperture radius.
            transmission: Intensity scaling applied to transmitted ray.
        """
        super().__init__(origin, **kwargs)
        self.focal_length = focal_length
        self.transmission = transmission
        #
        self.radius = radius
        self.surface = Circle(radius)
        #
        self._edge_color = "purple"

    def interact_local(self, ray):
        """Apply thin-lens deflection and Gaussian ``q`` propagation locally."""
        normal = np.array([1, 0, 0])  # normal in local frame is always x-axis
        P, t = self.intersect_point_local(ray)
        if ray.qo is None:
            qo = None
        else:
            q1 = ray.q_at_z(t)
            qo = q1 / (1 - (q1 / self.focal_length))
        #
        v0 = ray.direction
        f = self.focal_length
        # lens equation: v' = v - P/f
        v = v0 - P / f
        deflected_ray = ray.copy(
            origin=P, direction=v, intensity=ray.intensity * self.transmission, qo=qo
        )
        rays = [deflected_ray]
        return rays

    def render(self, ax, type: str, **kwargs):
        """Render lens outline."""
        return super().render(ax, type, color=self._edge_color, **kwargs)

    def get_bbox_local(self):
        """Return local bounding box from circular aperture geometry."""
        return self.surface.get_bbox_local()


class CylMirror(BaseMirror):
    """Cylindrical mirror segment."""

    def __init__(
        self,
        origin,
        radius: float = 0.5,
        height: float = 1.0,
        theta_range=(-np.pi, np.pi),
        **kwargs,
    ):
        """Initialize a cylindrical mirror."""
        super().__init__(origin, **kwargs)
        self.radius = radius
        self.height = height
        self.surface = Cylinder(radius, height, theta_range)
