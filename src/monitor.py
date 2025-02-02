from optical_component import *


class Monitor(OpticalComponent):
    def __init__(self, origin, width, height, **kwargs):
        super().__init__(origin, **kwargs)
        self.width = width
        self.height = height
        self.surface = Rectangle(width, height)
        self._edge_color = "orange"
        self.data = []  # list of tuples (P, intensity)

    @property
    def ndata(self):
        return len(self.data)

    def interact_local(self, ray):
        return [ray]

    def render(self, ax, type: str, **kwargs):
        return super().render(ax, type, color=self._edge_color, **kwargs)

    def get_bbox(self):
        return self.surface.get_bbox()

    def record(self, rays: List[Ray]):
        Pts = []
        for r in rays:
            local_ray = self.to_local_coordinates(r)
            P, t = self.intersect_point_local(local_ray)
            if P is not None:
                spot_size = r.spot_size(t) if r.qo is not None else 0
                Pts.append((P, local_ray.intensity, spot_size))
        self.data.extend(Pts)

    def _get_hist_y(self):
        yList = [data[0][1] for data in self.data]
        counts, bins = np.histogram(
            yList, bins=30, range=(-self.width / 2, self.width / 2)
        )
        return counts, bins

    @property
    def sum_intensity(self):
        return np.sum([data[1] for data in self.data])

    @property
    def avg_intensity(self):
        return np.mean([data[1] for data in self.data])

    @property
    def std_histy(self):
        counts, bins = self._get_hist_y()
        Ix = np.sum(counts * bins[:-1]) / np.sum(counts)
        Ixx = np.sum(counts * bins[:-1] ** 2) / np.sum(counts)
        return np.sqrt(Ixx - Ix**2)

    def render_hist(self, ax, type="Y", **kwargs):
        if type == "Y":
            counts, bins = self._get_hist_y()
            ax.bar(bins[:-1], counts, width=(bins[1] - bins[0]), **kwargs)
            ax.set_xlabel("Y")
        elif type == "YZ":
            yList = [data[0][1] for data in self.data]
            zList = [data[0][2] for data in self.data]
            ax.hist2d(
                yList,
                zList,
                bins=30,
                density=False,
                range=(
                    (-self.width / 2, self.width / 2),
                    (-self.height / 2, self.height / 2),
                ),
                **kwargs,
            )
            ax.set_xlabel("Y")
            ax.set_ylabel("Z")

    def render_scatter(self, ax, **kwargs):
        if len(self.data) == 0:
            return
        yList = np.array([data[0][1] for data in self.data])
        zList = np.array([data[0][2] for data in self.data])
        IList = np.array([data[1] for data in self.data])
        alpha = np.clip(IList, 0.1, 1)
        ax.scatter(yList, zList, marker="+", alpha=alpha, c="blue")
        spot_size_scale = kwargs.get("spot_size_scale", 1.0)
        spotsizeList = np.array([data[2] for data in self.data]) * spot_size_scale
        for spotsize, y, z, a in zip(spotsizeList, yList, zList, alpha):
            circle = plt.Circle((y, z), spotsize, color="red", alpha=a / 2, ec=None)
            ax.add_artist(circle)
        ax.set_xlim(-self.width / 2, self.width / 2)
        ax.set_ylim(-self.height / 2, self.height / 2)
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.set_title(str(self.name))
        ax.set_aspect("auto")
