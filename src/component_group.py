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
        self._bboxes = []
        self.components = []
        self.monitors = []
        self.rays = []
        self.refpoints = []
        self.name = kwargs.get("name", None)

    def __repr__(self):
        return f"ComponentGroup(origin={self.origin}, transform_matrix={self.transform_matrix})"

    @property
    def bbox(self):
        if self._bbox == (None, None, None, None, None, None):
            self._bbox = tuple(self.get_bbox())
        return tuple(self._bbox)

    @property
    def bboxes(self):
        if not self._bboxes:
            self.get_bboxes()
        return self._bboxes

    def get_bboxes(self) -> List[tuple]:
        self._bboxes = [c.bbox for c in self.components]
        return self._bboxes

    def get_bbox(self) -> tuple:
        self.get_bboxes()
        bbox_merged = self.surface.merge_bboxs(self._bboxes)
        return bbox_merged

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
        for point in self.refpoints:
            lp = -(point.origin - (old_origin + localpoint))
            point._RotAroundLocal(axis, lp, theta)

        return self

    def _RotAroundCenter(self, axis, theta):
        return self._RotAroundLocal(axis, [0, 0, 0], theta)

    def _Translate(self, movement):
        self.origin += np.array(movement)
        for ray in self.rays:
            ray._Translate(movement)
        for component in self.components:
            component._Translate(movement)
        for monitor in self.monitors:
            monitor._Translate(movement)
        for point in self.refpoints:
            point._Translate(movement)
        return self

    def render(self, ax, type: str, **kwargs):
        for component in self.components:
            component.render(ax, type, **kwargs)
        for point in self.refpoints:
            point.render(ax, type, **kwargs)

    def interact(self, ray: Ray) -> Union[Tuple[float, List[Ray]], Tuple[None, None]]:
        tList = []
        new_rays_list = []
        # first does this ray even intersect with the bbox of this component group
        EPS = 1e-9
        _, _, hits = solve_crosssection_ray_bboxes(ray.origin, ray.direction, self.bbox)
        if not hits[0]:
            return None, None
        # then check each component
        bboxes = [np.array(c.bbox) for c in self.components]
        t1s, t2s, hits = solve_crosssection_ray_bboxes(
            ray.origin, ray.direction, bboxes
        )
        # print(f"ComponentGroup.interact: ray {ray._id} hits {np.sum(hits)} components")
        hits_idx_mask = np.where(hits)[0]
        for i in hits_idx_mask:
            component = self.components[i]
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
        if hasattr(component, "rays"):
            self.rays.extend(component.rays)

    def add_components(self, components):
        self.components.extend(components)
        for component in components:
            if hasattr(component, "rays"):
                self.rays.extend(component.rays)

    def add_monitor(self, monitor):
        self.monitors.append(monitor)

    def add_monitors(self, monitors):
        self.monitors.extend(monitors)

    def add_refpoint(self, point):
        self.refpoints.append(point)


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


class MMA(ComponentGroup):
    """Microlens Mirror Array"""

    def __init__(self, origin, N, pitch, roc, n, thickness, roc_drift=0, **kwargs):
        super().__init__(origin, **kwargs)
        self.pitch = pitch
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
                roci = roc * (1 + roc_drift * np.random.randn())
                print(kwargs.get("transmission", None))
                mirror = SphereRefractive(
                    origin=o + [-roci, 0, 0],
                    radius=roci,
                    height=roci - np.sqrt(roci**2 - (self.pitch / 2) ** 2),
                    n1=n,
                    n2=1.0,
                    **kwargs,
                )
                self.add_component(mirror)
        #
        backsurface = SquareRefractive(
            origin=self.origin + np.array([thickness, 0, 0]),
            width=ny * pitch,
            height=nz * pitch,
            n1=1,
            n2=n,
            reflectivity=0,
            transmission=1,
        )
        self.add_component(backsurface)


class DMD(ComponentGroup):
    """Digital Micromirror Device"""

    def __init__(self, origin, N, pitch, tilt_angle=np.pi / 4, **kwargs):
        super().__init__(origin)
        self.pitch = pitch
        self.tilt_angle = tilt_angle
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
                mirror = SquareMirror(
                    origin=o,
                    width=pitch,
                    height=pitch,
                    reflectivity=1.0,
                    **kwargs,
                ).RotZ(self.tilt_angle)
                self.add_component(mirror)


class WedgePlate(ComponentGroup):
    def __init__(
        self,
        origin,
        width=1.0,
        height=1.0,
        thickness=1.0,
        wedge_angle=0.0,
        n1=1.0,
        n2=1.5,
        reflectivity=0,
        transmission=1,
        **kwargs,
    ):
        super().__init__(origin, **kwargs)
        self.add_component(
            SquareRefractive(
                origin + np.array([thickness / 2, 0, 0]),
                width,
                height,
                n1,
                n2,
                reflectivity=reflectivity,
                transmission=transmission,
                **kwargs,
            ).RotZ(wedge_angle / 2)
        )
        self.add_component(
            SquareRefractive(
                origin + np.array([-thickness / 2, 0, 0]),
                width,
                height,
                n2,
                n1,
                reflectivity=reflectivity,
                transmission=transmission,
                **kwargs,
            ).RotZ(-wedge_angle / 2)
        )


class MirrorPrism(ComponentGroup):
    def __init__(
        self,
        origin,
        width=1.0,
        height=1.0,
        reflectivity=1.0,
        **kwargs,
    ):
        """L=width, H=height,(L,L,sqrt(2)L)x H, origin at right-angle corner"""
        super().__init__(origin, **kwargs)
        self.add_component(
            SquareMirror(
                origin + np.array([0, width / 2, 0]),
                width,
                height,
                reflectivity=reflectivity,
                **kwargs,
            )._RotAroundLocal([0, 0, 1], [0, -width / 2, 0], np.pi / 4)
        )
        self.add_component(
            SquareMirror(
                origin + np.array([0, -width / 2, 0]),
                width,
                height,
                reflectivity=reflectivity,
                **kwargs,
            )._RotAroundLocal([0, 0, 1], [0, width / 2, 0], -np.pi / 4)
        )


class MirrorCube(ComponentGroup):
    def __init__(
        self,
        origin,
        L=1.0,
        reflectivity=1.0,
        **kwargs,
    ):
        """Corner cube with the x-axis forming equal angles to all three mirrors."""
        super().__init__(origin, **kwargs)
        mirror_x = SquareMirror(
            self.origin + np.array([0.0, L / 2, L / 2]),
            width=L,
            height=L,
            reflectivity=reflectivity,
            **kwargs,
        )
        mirror_y = SquareMirror(
            self.origin + np.array([L / 2, 0.0, L / 2]),
            width=L,
            height=L,
            reflectivity=reflectivity,
            **kwargs,
        ).RotZ(np.pi / 2)
        mirror_z = SquareMirror(
            self.origin + np.array([L / 2, L / 2, 0.0]),
            width=L,
            height=L,
            reflectivity=reflectivity,
            **kwargs,
        ).RotY(-np.pi / 2)
        self.add_components([mirror_x, mirror_y, mirror_z])
        self._RotAroundLocal([0, 0, 1], [0, 0, 0], -np.pi / 4)
        self._RotAroundLocal([0, 1, 0], [0, 0, 0], np.arccos(np.sqrt(2 / 3)))


class DovePrism(ComponentGroup):
    def __init__(self, origin, L, D, Ng, **kwargs):
        """
        L: length of the base
        D: height of the prism
        Ng: refractive index of the prism material
        """
        super().__init__(origin, **kwargs)
        self.L = L
        self.D = D
        self.Ng = Ng

        Lu = L - 2 * D

        # 2) Arbitrary 3-D triangle
        verts2d = np.array(
            [
                [-L / 2, 0],
                [L / 2, 0],
                [Lu / 2, D],
                [-Lu / 2, D],
            ]
        )
        poly2d = Polygon(verts2d)

        SL = BaseRefraciveSurface(origin=[-D / 2, 0, 0], n1=Ng, n2=1)
        SL.surface = poly2d
        SR = BaseRefraciveSurface(origin=[D / 2, 0, 0], n1=1, n2=Ng)
        SR.surface = poly2d
        SU = SquareRefractive(origin=[0, 0, D], width=Lu, height=D, n1=Ng, n2=1).RotY(
            np.pi / 2
        )
        SB = SquareRefractive(origin=[0, 0, 0], width=L, height=D, n1=1, n2=Ng).RotY(
            np.pi / 2
        )
        verts3dI = np.array(
            [
                [-D / 2, -L / 2, 0],
                [D / 2, -L / 2, 0],
                [D / 2, -Lu / 2, D],
                [-D / 2, -Lu / 2, D],
            ]
        )
        verts3dO = np.array(
            [
                [-D / 2, L / 2, 0],
                [D / 2, L / 2, 0],
                [D / 2, Lu / 2, D],
                [-D / 2, Lu / 2, D],
            ]
        )
        poly3dI = Polygon(verts3dI)
        poly3dO = Polygon(verts3dO)
        SI = BaseRefraciveSurface(origin=[0, 0, 0], n1=Ng, n2=1)
        SI.surface = poly3dI
        SO = BaseRefraciveSurface(origin=[0, 0, 0], n1=1, n2=Ng)
        SO.surface = poly3dO

        self.add_components([SL, SR, SU, SB, SI, SO])

    @property
    def z0(self):
        """the height of rays that will go straight through the prism"""
        theta = np.arcsin((np.sqrt(2) / 2) / self.Ng)
        return (self.L / 2) / (1 + np.tan(np.pi / 4 + theta))


class PlanoConvexLens(ComponentGroup):
    def __init__(self, origin, EFL, CT, diameter, R, **kwargs):
        super().__init__(origin, **kwargs)
        # calculate refractive index from EFL and R
        n = 1 + R / EFL
        o1 = np.array([-R + CT, 0, 0]) + self.origin
        D = CT / n  # principal plane location from flat surface
        oD = np.array([-D, 0, 0])
        height_curve = R - np.sqrt(R**2 - (diameter / 2) ** 2)
        curved_face = SphereRefractive(
            origin=o1 + oD, radius=R, height=height_curve, n1=1.0, n2=n, **kwargs
        )
        self.add_component(curved_face)
        flat_face = CircleRefractive(
            origin=self.origin + oD, radius=diameter / 2, n1=n, n2=1.0, **kwargs
        )
        self.add_component(flat_face)


class BiConvexLens(ComponentGroup):
    def __init__(self, origin, R1, R2, CT, diameter, n, **kwargs):
        super().__init__(origin, **kwargs)
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
