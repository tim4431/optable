import scipy
from .base import *
from .ray import *
from .surfaces import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class OpticalComponent(Vector):
    def __init__(self, origin, **kwargs):
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

    def __repr__(self):
        return f"OpticalComponent(origin={self.origin}, transform_matrix=\n{self.transform_matrix})"

    # normal in local fram is always x-axis
    @property
    def normal(self):
        return self.transform_matrix @ np.array([1, 0, 0])

    @property
    def tangent(self):
        return self.transform_matrix @ np.array([0, 1, 0])

    @property
    def bbox(self):
        self._bbox = tuple(self.get_bbox())
        return tuple(self._bbox)

    def get_bbox(self) -> tuple:
        raise NotImplementedError("Subclasses must implement get_bbox method")

    def _RotAroundLocal(self, axis, localpoint, theta):
        R = self.R(axis, theta)
        localpoint = np.array(localpoint)
        self.transform_matrix = np.dot(R, self.transform_matrix)
        self.origin = self.origin + np.dot(R, -localpoint) + localpoint
        return self

    def _RotAroundCenter(self, axis, theta):
        return self._RotAroundLocal(axis, [0, 0, 0], theta)

    def to_local_coordinates(self, ray: Ray) -> Ray:
        """
        Transforms the ray to the local coordinate system of the optical component.
        """
        R = np.linalg.inv(self.transform_matrix)
        local_origin = np.dot(R, ray.origin - self.origin)
        local_direction = np.dot(R, ray.direction)
        return ray.copy(origin=local_origin, direction=local_direction)

    def to_lab_coordinates(self, ray: Ray) -> Ray:
        """
        Transforms the ray from local coordinates back to the lab coordinate system.
        """
        R = self.transform_matrix
        global_origin = np.dot(R, ray.origin) + self.origin
        global_direction = np.dot(R, ray.direction)
        return ray.copy(origin=global_origin, direction=global_direction)

    def _find_zero(self, f, a, b, num_intervals: int = 100) -> np.ndarray:
        t_values = np.linspace(a, b, int(num_intervals))
        sols = []
        for i in range(len(t_values) - 1):
            t_start, t_end = t_values[i], t_values[i + 1]
            if f(t_start) * f(t_end) < 0:
                t = scipy.optimize.brentq(f, t_start, t_end)
                sols.append(t)
        return np.array(sols)

    def intersect_point_local(
        self, ray: Ray
    ) -> Union[Tuple[np.ndarray, float], Tuple[None, None]]:
        """
        (In local coordinates)
        Calculates the intersection point between the ray and the optical component.
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

            tList = self._find_zero(f, 0, 1e2)
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
                for t in np.sort(tvalid):
                    P = ray.origin + t * ray.direction
                    if self.surface.within_boundary(P):
                        return P, t
                return None, None

    def interact_local(self, ray: Ray) -> List[Ray]:
        """
        Interacts with the optical component in the local coordinate system.
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
        """
        Render the optical component.

        Parameters:
            ax: Matplotlib 2D axis object for plotting.
            type: str, the type of rendering (e.g., "Z" for 2D, "3D" for 3D).
            kwargs: Additional arguments for customization (e.g., edge color).
        """
        if not self.render_obj:
            return
        # Get edge color and line width from kwargs
        color = kwargs.get("color", "black")
        linewidth = kwargs.get("linewidth", 2)
        global_x, global_y, global_z = self._get_boundary_points(type)
        detailed_render = kwargs.get("detailed_render", False)
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
                    scale=2,
                    scale_units="xy",
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
                    scale=2,
                    scale_units="xy",
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
        """
        Interacts with the optical component.

        Returns:
            t: float, the distance between the ray origin and the intersection point.
            rays: List[Ray], the new rays after interaction.
        """
        if not ray.alive:
            return None, None
        else:
            local_ray = self.to_local_coordinates(ray)
            P, t = self.intersect_point_local(local_ray)
            if P is None:
                return None, None
            else:
                truncted_ray = ray.copy(length=t, alive=False)
                # new rays after interaction
                local_rays_after_interaction = self.interact_local(
                    local_ray
                )  # List[Ray]
                lab_rays_after_interaction = [
                    self.to_lab_coordinates(local_ray)
                    for local_ray in local_rays_after_interaction
                ]
                return t, [truncted_ray] + lab_rays_after_interaction

    def patch_block(self, width, height):
        obj = Block(self.origin, hole=self.surface, width=width, height=height)
        obj.transform_matrix = self.transform_matrix
        return obj

    def gather_components(
        self, avoid_flatten_classname: List = [], ignore_classname: List = []
    ) -> List:
        """
        Gather all components in the optical component.
        """

        def _repr_self_dict():
            d = {
                "class": self.__class__.__name__,
                "origin_x": self.origin[0],
                "origin_y": self.origin[1],
                "origin_z": self.origin[2],
                "normal_x": self.normal[0],
                "normal_y": self.normal[1],
                "normal_z": self.normal[2],
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


class Nothing(OpticalComponent):
    def __init__(self, origin, **kwargs):
        super().__init__(origin, **kwargs)
        self.surface = Plane()
        self._edge_color = "black"

    def interact_local(self, ray):
        return [ray]  # every ray is transmitted

    def render(self, ax, type: str, **kwargs):
        super().render(ax, type, color=self._edge_color, **kwargs)

    def get_bbox(self):
        return self.surface.get_bbox()


class Block(OpticalComponent):
    def __init__(
        self,
        origin,
        hole: Union[Surface | None] = None,
        width: float = 1.0,
        height: float = 1.0,
        **kwargs,
    ):
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
        return []  # every ray is absorbed

    def render(self, ax, type: str, **kwargs):
        super().render(ax, type, color=self._edge_color, **kwargs)

    def get_bbox(self):
        return self.surface.get_bbox()


class BaseMirror(OpticalComponent):
    def __init__(
        self,
        origin,
        reflectivity: float = 1.0,
        transmission: float = 0.0,
        **kwargs,
    ):
        super().__init__(origin, **kwargs)
        self.reflectivity = reflectivity
        self.transmission = transmission
        #
        self._edge_color = "green"

    def interact_local(self, ray):
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
            )
            rays.append(reflected_ray)
        if self.transmission > 0:
            transmitted_ray = ray.copy(
                origin=P,
                direction=ray.direction,
                intensity=ray.intensity * self.transmission,
                qo=qo,
            )
            rays.append(transmitted_ray)
        #
        return rays

    def render(self, ax, type: str, **kwargs):
        super().render(ax, type, color=self._edge_color, **kwargs)

    def get_bbox(self):
        return self.surface.get_bbox()


class BaseRefraciveSurface(OpticalComponent):
    def __init__(
        self,
        origin,
        n1: Union[float, callable] = 1.0,
        n2: Union[float, callable] = 1.0,
        reflectivity: float = 0.0,
        transmission: float = 1.0,
        **kwargs,
    ):
        """
        n1: x>0, n2: x<0
        """
        super().__init__(origin, **kwargs)
        if isinstance(n1, (int, float)):
            self.n1 = lambda l: n1
        else:
            self.n1 = n1
        if isinstance(n2, (int, float)):
            self.n2 = lambda l: n2
        else:
            self.n2 = n2
        #
        self.reflectivity = reflectivity
        self.transmission = transmission
        #
        self._edge_color = "gray"

    def interact_local(self, ray):
        P, t = self.intersect_point_local(ray)
        normal = self.surface.normal(P)
        n1, n2 = self.n1(ray.wavelength), self.n2(ray.wavelength)
        #
        rays = []
        # take n2 to n1 as normal>0
        if np.dot(ray.direction, normal) < 0:  # incident from n1 to n2
            normal = -normal
            n1, n2 = n2, n1
        # now we are always treating the case incident from n2 to n1
        qo = None if ray.qo is None else ray.q_at_z(t) * n1 / n2
        r_n = np.dot(ray.direction, normal)
        r_t = ray.direction - r_n * normal
        cos_theta_i = r_n
        sin_theta_i = np.sqrt(1 - cos_theta_i**2)
        sin_theta_t = (n2 * sin_theta_i) / n1
        if sin_theta_t < 1:
            cos_theta_t = np.sqrt(1 - sin_theta_t**2)
            transmitted_direction = (n2 / n1) * r_t + cos_theta_t * normal
            transmitted_ray = ray.copy(
                origin=P,
                direction=transmitted_direction,
                intensity=ray.intensity * self.transmission,
                qo=qo,
                n=n1,
            )
            rays.append(transmitted_ray)
        else:
            # total internal reflection
            normal = -normal
            reflected_direction = ray.direction + 2 * cos_theta_i * normal
            reflected_ray = ray.copy(
                origin=P,
                direction=reflected_direction,
                intensity=ray.intensity,
                qo=qo,
            )
            rays.append(reflected_ray)

        #
        if self.reflectivity > 0:
            normal = -normal
            reflected_direction = ray.direction + 2 * cos_theta_i * normal
            reflected_ray = ray.copy(
                origin=P,
                direction=reflected_direction,
                intensity=ray.intensity * self.reflectivity,
                qo=qo,
            )
            rays.append(reflected_ray)
        #
        return rays

    def render(self, ax, type: str, **kwargs):
        super().render(ax, type, color=self._edge_color, **kwargs)

    def get_bbox(self):
        return self.surface.get_bbox()


class Mirror(BaseMirror):
    def __init__(
        self,
        origin,
        radius: float = 0.5,
        reflectivity: float = 1.0,
        transmission: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            origin, reflectivity=reflectivity, transmission=transmission, **kwargs
        )
        self.radius = radius
        self.surface = Circle(radius)


class SquareMirror(BaseMirror):
    def __init__(
        self,
        origin,
        width: float = 1.0,
        height: float = 1.0,
        reflectivity: float = 1.0,
        transmission: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            origin, reflectivity=reflectivity, transmission=transmission, **kwargs
        )
        self.width = width
        self.height = height
        self.surface = Rectangle(width, height)


class SquareRefractive(BaseRefraciveSurface):
    def __init__(
        self,
        origin,
        width: float = 1.0,
        height: float = 1.0,
        n1: Union[float, callable] = 1.0,
        n2: Union[float, callable] = 1.0,
        reflectivity: float = 0.0,
        transmission: float = 1.0,
        **kwargs,
    ):
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
    def __init__(
        self,
        origin,
        radius: float = 0.5,
        n1: Union[float, callable] = 1.0,
        n2: Union[float, callable] = 1.0,
        reflectivity: float = 0.0,
        transmission: float = 1.0,
        **kwargs,
    ):
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
    def __init__(
        self,
        origin,
        radius: float = 0.5,
        height: float = 0.5,
        n1: Union[float, callable] = 1.0,
        n2: Union[float, callable] = 1.0,
        reflectivity: float = 0.0,
        transmission: float = 1.0,
        **kwargs,
    ):
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
        self.surface = Sphere(radius, height)


class BeamSplitter(Mirror):
    def __init__(self, origin, radius: float = 0.5, eta: float = 0.5, **kwargs):
        reflectivity = np.sqrt(eta)
        transmission = np.sqrt(1 - eta)
        super().__init__(
            origin,
            radius=radius,
            reflectivity=reflectivity,
            transmission=transmission,
            **kwargs,
        )


class Lens(OpticalComponent):
    def __init__(
        self,
        origin,
        focal_length,
        radius: float = 0.5,
        transmission: float = 1.0,
        **kwargs,
    ):
        super().__init__(origin, **kwargs)
        self.focal_length = focal_length
        self.transmission = transmission
        #
        self.radius = radius
        self.surface = Circle(radius)
        #
        self._edge_color = "purple"

    def interact_local(self, ray):
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
        # deflected_ray = Ray(P, v, ray.intensity * self.transmission)
        deflected_ray = ray.copy(origin=P, direction=v, intensity=ray.intensity, qo=qo)
        rays = [deflected_ray]
        return rays

    def render(self, ax, type: str, **kwargs):
        return super().render(ax, type, color=self._edge_color, **kwargs)

    def get_bbox(self):
        return self.surface.get_bbox()


class CylMirror(BaseMirror):
    def __init__(
        self,
        origin,
        radius: float = 0.5,
        height: float = 1.0,
        theta_range=(-np.pi, np.pi),
        **kwargs,
    ):
        super().__init__(origin, **kwargs)
        self.radius = radius
        self.height = height
        self.surface = Cylinder(radius, height, theta_range)
