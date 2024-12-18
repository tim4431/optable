from optical_component import *


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

    def __repr__(self):
        return f"ComponentGroup(origin={self.origin}, transform_matrix={self.transform_matrix})"

    def _RotAroundLocal(self, axis, localpoint, theta):
        R = self.R(axis, theta)
        localpoint = np.array(localpoint)
        self.transform_matrix = np.dot(R, self.transform_matrix)
        old_origin = self.origin
        self.origin = self.origin + np.dot(R, -localpoint) + localpoint
        #
        for component in self.components:
            lp = -(component.origin - old_origin)
            component._RotAroundLocal(axis, lp, theta)
        return self

    def _RotAroundCenter(self, axis, theta):
        return self._RotAroundLocal(axis, [0, 0, 0], theta)

    def _Translate(self, direction, distance):
        self.origin += np.array(direction) * distance
        for component in self.components:
            component._Translate(direction, distance)
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


class GlassSlab(ComponentGroup):
    def __init__(
        self,
        origin,
        width,
        height,
        thickness,
        n1,
        n2,
        reflectivity=0,
        transmission=0,
    ):
        super().__init__(origin)
        self.components.append(
            SquareRefractive(
                origin + np.array([+thickness / 2, 0, 0]),
                width,
                height,
                n1,
                n2,
                reflectivity=reflectivity,
                transmission=transmission,
            )
        )
        self.components.append(
            SquareRefractive(
                origin + np.array([-thickness / 2, 0, 0]),
                width,
                height,
                n2,
                n1,
                reflectivity=reflectivity,
                transmission=transmission,
            )
        )


def MLA(o, pitch, f, r, N) -> List[Lens]:
    """
    Micro-lens array
    - o: origin of the first lens
    - pitch: pitch
    - f: focal length
    - r: radius of the lens
    - N: number of lenses (int/tuple)
    """


class MLA(ComponentGroup):
    def __init__(self, origin, N, pitch, focal_length, radius):
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
                z = i * pitch
                y = j * pitch
                o = np.array([0, y, z]) + self.origin
                lens = Lens(o, focal_length=focal_length, radius=radius)
                self.components.append(lens)
