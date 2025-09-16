from .optical_component import *


class ComponentGroup(OpticalComponent):
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
        self.components = []
        self.monitors = []
        self.rays = []

    def __repr__(self):
        return f"ComponentGroup(origin={self.origin}, transform_matrix={self.transform_matrix})"

    def _RotAroundLocal(self, axis, localpoint, theta):
        # Note that localpoint is in the local coord, and origins are all in lab coord
        R = self.R(axis, theta)
        localpoint = np.array(localpoint)
        self.transform_matrix = np.dot(R, self.transform_matrix)
        old_origin = self.origin
        self.origin = self.origin + np.dot(R, -localpoint) + localpoint
        #
        for ray in self.rays:
            lp = -(ray.origin - (old_origin + localpoint))
            ray._RotAroundLocal(axis, lp, theta)
        for component in self.components:
            lp = -(component.origin - (old_origin + localpoint))
            component._RotAroundLocal(axis, lp, theta)
        for monitor in self.monitors:
            lp = -(monitor.origin - (old_origin + localpoint))
            monitor._RotAroundLocal(axis, lp, theta)
        return self

    def _RotAroundCenter(self, axis, theta):
        return self._RotAroundLocal(axis, [0, 0, 0], theta)

    def _Translate(self, direction, distance):
        self.origin += np.array(direction) * distance
        for ray in self.rays:
            ray._Translate(direction, distance)
        for component in self.components:
            component._Translate(direction, distance)
        for monitor in self.monitors:
            monitor._Translate(direction, distance)
        return self

    def render(self, ax, type: str, **kwargs):
        for component in self.components:
            component.render(ax, type, **kwargs)

    def interact(self, ray: Ray) -> Union[Tuple[float, List[Ray]], Tuple[None, None]]:
        tList = []
        new_rays_list = []
        for component in self.components:
            t, new_rays = component.interact(ray)
            if t is not None:
                tList.append(t)
                new_rays_list.append(new_rays)
        #
        # Find the closest intersection
        if len(tList) > 0:
            idx = np.argmin(tList)
            return tList[idx], new_rays_list[idx]
        else:
            return None, None

    def add_rays(self, rays):
        self.rays.extend(rays)

    def add_component(self, component):
        self.components.append(component)

    def add_components(self, components):
        self.components.extend(components)

    def add_monitor(self, monitor):
        self.monitors.append(monitor)

    def add_monitors(self, monitors):
        self.monitors.extend(monitors)


class GlassSlab(ComponentGroup):
    def __init__(
        self,
        origin,
        width=1.0,
        height=1.0,
        thickness=1.0,
        n1=1.0,
        n2=1.5,
        reflectivity=0,
        transmission=1,
        **kwargs,
    ):
        super().__init__(origin, **kwargs)
        self.add_component(
            SquareRefractive(
                origin + np.array([0, 0, 0]),
                width,
                height,
                n1,
                n2,
                reflectivity=reflectivity,
                transmission=transmission,
                **kwargs,
            )
        )
        self.add_component(
            SquareRefractive(
                origin + np.array([-thickness, 0, 0]),
                width,
                height,
                n2,
                n1,
                reflectivity=reflectivity,
                transmission=transmission,
                **kwargs,
            )
        )


class CircleGlassSlab(ComponentGroup):
    def __init__(
        self,
        origin,
        radius=1.0,
        thickness=1.0,
        n1=1.0,
        n2=1.5,
        reflectivity1=0,
        transmission1=1,
        reflectivity2=0,
        transmission2=1,
        **kwargs,
    ):
        super().__init__(origin, **kwargs)
        self.radius = radius
        self.add_component(
            CircleRefractive(
                origin + np.array([0, 0, 0]),
                radius,
                n1,
                n2,
                reflectivity=reflectivity1,
                transmission=transmission1,
                **kwargs,
            )
        )
        self.add_component(
            CircleRefractive(
                origin + np.array([-thickness, 0, 0]),
                radius,
                n2,
                n1,
                reflectivity=reflectivity2,
                transmission=transmission2,
                **kwargs,
            )
        )


class MLA(ComponentGroup):
    def __init__(self, origin, N, pitch, focal_length, radius, focal_drift=0, **kwargs):
        super().__init__(origin)
        self.pitch = pitch
        self.focal_length = focal_length
        self.radius = radius
        #
        if isinstance(N, int):
            N = (N, 1)
        ny, nz = N
        #
        for i in range(nz):
            for j in range(ny):
                z = (i - (nz - 1) / 2) * pitch
                y = (j - (ny - 1) / 2) * pitch
                o = np.array([0, y, z]) + self.origin
                f = focal_length * (1 + focal_drift * np.random.randn())
                lens = Lens(origin=o, focal_length=f, radius=radius, **kwargs)
                self.add_component(lens)


class PlanoConvexLens(ComponentGroup):
    def __init__(self, origin, EFL, CT, diameter, R, **kwargs):
        super().__init__(origin)
        # calculate refractive index from EFL and R
        n = 1 + R / EFL
        o1 = np.array([-R + CT, 0, 0]) + self.origin
        height_curve = R - np.sqrt(R**2 - (diameter / 2) ** 2)
        curved_face = SphereRefractive(
            origin=o1, radius=R, height=height_curve, n1=1.0, n2=n, **kwargs
        )
        self.add_component(curved_face)
        flat_face = CircleRefractive(
            origin=self.origin, radius=diameter / 2, n1=n, n2=1.0, **kwargs
        )
        self.add_component(flat_face)


class BiConvexLens(ComponentGroup):
    def __init__(self, origin, R1, R2, CT, diameter, n, **kwargs):
        super().__init__(origin)
        o1 = np.array([R1, 0, 0]) + self.origin
        o2 = np.array([-R2 + CT, 0, 0]) + self.origin
        height_curve1 = R1 - np.sqrt(R1**2 - (diameter / 2) ** 2)
        height_curve2 = R2 - np.sqrt(R2**2 - (diameter / 2) ** 2)
        curved_face1 = SphereRefractive(
            origin=o1, radius=R1, height=height_curve1, n1=1.0, n2=n, **kwargs
        ).RotZ(np.pi)
        curved_face2 = SphereRefractive(
            origin=o2,
            radius=R2,
            height=height_curve2,
            n1=1.0,
            n2=n,
            **kwargs,
        )
        self.add_component(curved_face1)
        self.add_component(curved_face2)
