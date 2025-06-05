from .optical_component import *


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
                # spot_size = r.spot_size(t) if r.qo is not None else 0
                # Pts.append((P, local_ray.intensity, spot_size))
                Pts.append((P, local_ray.intensity, t, r))
        self.data.extend(Pts)

    def _get_hist_y(self):
        yList = [data[0][1] for data in self.data]
        counts, bins = np.histogram(
            yList, bins=30, range=(-self.width / 2, self.width / 2)
        )
        return counts, bins

    def get_waist_distance(self):
        tList = np.array([data[2] for data in self.data])
        rList = np.array([data[3] for data in self.data])
        # waist_distance_List_raw = np.array(
        #     [r.distance_to_waist(r.q_at_z(t)) for r, t in zip(rList, tList)]
        # )
        # account for beam propagation direction
        waist_distance_List = []
        for r, t in zip(rList, tList):
            q = r.q_at_z(t)
            distance_raw = r.distance_to_waist(q)
            if np.dot(r.direction, self.normal) > 0:
                # if the ray is propagating towards the monitor, we take the negative distance
                distance_raw = -distance_raw
            waist_distance_List.append(distance_raw)
        waist_distance_List = np.array(waist_distance_List)

        return waist_distance_List

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
        if hasattr(self, "name") and self.name is not None:
            print(f"Rendering {self.name} with {self.ndata} data points")
        if len(self.data) == 0:
            return
        yList = np.array([data[0][1] for data in self.data])
        zList = np.array([data[0][2] for data in self.data])
        IList = np.array([data[1] for data in self.data])
        alpha = np.clip(IList, 0.1, 1)
        ax.scatter(yList, zList, marker="+", alpha=alpha, c="blue")
        #
        gaussian_beam = kwargs.get("gaussian_beam", False)
        if gaussian_beam:
            spot_size_scale = kwargs.get("spot_size_scale", 1.0)
            tList = np.array([data[2] for data in self.data])
            rList = np.array([data[3] for data in self.data])
            spotsizeList = (
                np.array([r.spot_size(t) for r, t in zip(rList, tList)])
                * spot_size_scale
            )
            waistList = (
                np.array([r.waist(r.q_at_z(t)) for r, t in zip(rList, tList)])
                * spot_size_scale
            )
            annote_spotsize = kwargs.get("annote_spotsize", False)
            annote_waist = kwargs.get("annote_waist", False)
            _sign = 1
            for spotsize, waist, y, z, a in zip(
                spotsizeList, waistList, yList, zList, alpha
            ):
                circle = plt.Circle((y, z), spotsize, color="red", alpha=a / 2, ec=None)
                ax.add_artist(circle)
                if annote_spotsize:
                    ax.text(
                        y,
                        z + 0.1 * self.height * _sign,
                        f"sp={spotsize:.2e}",
                        fontsize=8,
                        color="black",
                    )
                if annote_waist:
                    ax.text(
                        y,
                        z + 0.1 * self.height * _sign,
                        f"w={waist:.2e}",
                        fontsize=8,
                        color="black",
                    )
                _sign = (_sign + 1 + 7) % 7 - 3  # alternate sign for waist annotation
            annote_dis_std = kwargs.get("annote_dis_std", False)
            if annote_dis_std:
                waist_distance_List = self.get_waist_distance()
                # print(
                #     f"Waist distance: {waist_distance_List}, std={np.std(waist_distance_List):.4f}"
                # )
                std_distance = np.std(waist_distance_List)
                ax.text(0, self.height / 2 * 0.8, f"Std(z)={std_distance:.4f}")
            #
            idx_y = np.argsort(yList)
            y_sorted = yList[idx_y]
            dy = np.diff(y_sorted)
            std_y = np.std(dy)
            z_sorted = zList[idx_y]
            dz = np.diff(z_sorted)
            std_z = np.std(dz)
            #
            annote_pos_std = kwargs.get("annote_pos_std", False)
            annote_pos_mean = kwargs.get("annote_pos_mean", False)
            if annote_pos_std:
                # annote the std of scatter points in Y and Z
                ax.text(
                    0,
                    self.height / 2 * 0.6,
                    f"SX={std_y:.4f}, SY={std_z:.4f}",
                )
            if annote_pos_mean:
                # annote the mean of scatter points in Y and Z
                mean_dy = np.mean(dy)
                mean_dz = np.mean(dz)
                ax.text(
                    0,
                    self.height / 2 * 0.4,
                    f"MX={mean_dy:.4f}, MY={mean_dz:.4f}",
                )

        ax.set_xlim(-self.width / 2, self.width / 2)
        ax.set_ylim(-self.height / 2, self.height / 2)
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        if hasattr(self, "name") and self.name is not None:
            ax.set_title(str(self.name))
        ax.set_aspect("auto")
