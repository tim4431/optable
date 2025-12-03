from .optical_component import *


class Monitor(OpticalComponent):
    def __init__(self, origin, width, height, **kwargs):
        super().__init__(origin, **kwargs)
        self.width = width
        self.height = height
        self.surface = Rectangle(width, height)
        self._edge_color = "orange"
        self._data_raw = []  # list of tuples (P, intensity)
        self._sortYZindex = None
        self._sorted_data = []
        self._updated = False
        self.name = kwargs.get("name", None)

    @property
    def ndata(self):
        return len(self._data_raw)

    @property
    def data(self):
        if not self._sorted_data or self._updated:
            self._sorted_data = [self._data_raw[i] for i in self.sortYZIndex]
            self._updated = False
        return self._sorted_data

    @property
    def rays(self):
        return [data[3] for data in self.data]

    @property
    def raw_yList(self):
        return np.array([data[0][1] for data in self._data_raw])

    @property
    def raw_zList(self):
        return np.array([data[0][2] for data in self._data_raw])

    @property
    def yList(self):
        return np.array([data[0][1] for data in self.data])

    @property
    def zList(self):
        return np.array([data[0][2] for data in self.data])

    @property
    def IList(self):
        """intensity list"""
        return np.array([data[1] for data in self.data])

    @property
    def tList(self):
        """tilt list"""
        return np.array([data[2] for data in self.data])

    @property
    def rList(self):
        """ray list"""
        return np.array([data[3] for data in self.data])

    @property
    def directionList(self):
        return np.array([r.direction for r in self.rList])

    @property
    def sortYZIndex(self):
        self._sortYZindex = np.lexsort((self.raw_zList, self.raw_yList))
        return self._sortYZindex

    @property
    def tYList(self):
        directionList = self.directionList  # n*3
        # y component of the direction vector is the second component
        tYList = directionList[:, 1]  # n
        return tYList

    @property
    def tZList(self):
        directionList = self.directionList  # n*3
        # z component of the direction vector is the third component
        tZList = directionList[:, 2]  # n
        return tZList

    def get_ray_i(self, idx):
        rayi = self.rays[idx]
        hity = self.yList[idx]
        hitz = self.zList[idx]
        tilty = self.tYList[idx]
        tiltz = self.tZList[idx]
        intensity = self.IList[idx]
        hit_point = [hity, hitz, tilty, tiltz, intensity]
        return rayi, hit_point

    def get_ray_id(self, ray_id):
        for idx, r in enumerate(self.rays):
            if r._id == ray_id:
                return self.get_ray_i(idx)
        return None, None

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
        self._data_raw.extend(Pts)
        self._updated = True

    def _get_hist_y(self):
        yList = self.yList
        counts, bins = np.histogram(
            yList, bins=30, range=(-self.width / 2, self.width / 2)
        )
        return counts, bins

    def get_waist_distance(self):
        tList = self.tList
        rList = self.rList
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

    def get_beam_waist(self):
        tList = self.tList
        rList = self.rList
        waist_List = []
        for r, t in zip(rList, tList):
            q = r.q_at_z(t)
            waist_List.append(r.waist(q))
        return np.array(waist_List)

    def get_delta_pos(self):
        yList = self.yList
        zList = self.zList
        if len(yList) == 0 or len(zList) == 0:
            return np.array([0.0]), np.array([0.0])

        idx_y = np.argsort(yList)
        y_sorted = yList[idx_y]
        dy = np.diff(y_sorted)
        z_sorted = zList[idx_y]
        dz = np.diff(z_sorted)
        return dy, dz

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

    def export_rays_npz(self, filename: str):
        print(f"Exporting {self.ndata} rays to {filename} ...")
        xList = self.yList
        yList = self.zList
        tXList = self.tYList
        tYList = self.tZList
        IList = self.IList
        np.savez(
            filename,
            xList=xList,
            yList=yList,
            tXList=tXList,
            tYList=tYList,
            IList=IList,
        )

    def render_hist(self, ax, type="Y", **kwargs):
        if type == "Y":
            counts, bins = self._get_hist_y()
            ax.bar(bins[:-1], counts, width=(bins[1] - bins[0]), **kwargs)
            ax.set_xlabel("Y")
        elif type == "YZ":
            ax.hist2d(
                self.yList,
                self.zList,
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
        if self.ndata == 0:
            return
        yList = self.yList
        zList = self.zList
        IList = self.IList
        alpha = np.clip(IList, 0.1, 1)
        ax.scatter(yList, zList, marker="+", alpha=alpha, c="blue")
        #
        annote_delta_pos = kwargs.get("annote_delta_pos", False)
        if annote_delta_pos:
            dy, dz = self.get_delta_pos()
            std_y = np.std(dy) if len(dy) > 0 else 0.0
            std_z = np.std(dz) if len(dz) > 0 else 0.0
            # annote the std of scatter points in Y and Z
            ax.text(
                0,
                self.height / 2 * 0.6,
                f"SX={std_y:.4f}, SY={std_z:.4f}",
            )
            # annote the mean of scatter points in Y and Z
            mean_dy = np.mean(dy) if len(dy) > 0 else 0.0
            mean_dz = np.mean(dz) if len(dz) > 0 else 0.0
            ax.text(
                0,
                self.height / 2 * 0.4,
                f"MX={mean_dy:.4f}, MY={mean_dz:.4f}",
            )
        #
        annote_yz_index = kwargs.get("annote_yz_index", False)
        if annote_yz_index:
            sort_index = self.sortYZIndex
            for i, (y, z) in enumerate(zip(yList[sort_index], zList[sort_index])):
                ax.text(
                    y,
                    z,
                    f"{i}",
                    fontsize=8,
                    color="black",
                )
        # GAUSSIAN BEAM
        gaussian_beam = kwargs.get("gaussian_beam", False)
        if gaussian_beam:
            spot_size_scale = kwargs.get("spot_size_scale", 1.0)
            tList = self.tList
            rList = self.rList
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
            #
            annote_dis_std = kwargs.get("annote_dis_std", False)
            if annote_dis_std:
                waist_distance_List = self.get_waist_distance()
                std_distance = np.std(waist_distance_List)
                ax.text(0, self.height / 2 * 0.8, f"Std(z)={std_distance:.4f}")
            #
            annote_waist_std = kwargs.get("annote_waist_std", False)
            if annote_waist_std:
                waist_List = self.get_beam_waist()
                std_waist = np.std(waist_List)
                mean_waist = np.mean(waist_List)
                ax.text(
                    0,
                    self.height / 2 * 0.2,
                    f"Std(w)={std_waist*1e4:.4f}, M(w)={mean_waist*1e4:.4f}",
                )
            #
            annote_beam_tilt = kwargs.get("annote_beam_tilt", False)
            if annote_beam_tilt:
                tYList = self.tYList
                tZList = self.tZList
                if len(tYList) > 0 and len(tZList) > 0:
                    std_tY = np.std(tYList)
                    std_tZ = np.std(tZList)
                    ax.text(
                        0,
                        self.height / 2 * 0.0,
                        f"Std(tY)={std_tY:.4f}, Std(tZ)={std_tZ:.4f}",
                    )

        ax.set_xlim(-self.width / 2, self.width / 2)
        ax.set_ylim(-self.height / 2, self.height / 2)
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        if hasattr(self, "name") and self.name is not None:
            ax.set_title(str(self.name))
        ax.set_aspect("auto")

    def render_delta_pos_y(self, ax, **kwargs):
        dy, dz = self.get_delta_pos()
        # bar plot of dy
        ax.bar(np.arange(len(dy)), dy, **kwargs)
        ax.set_ylim(np.min(dy) * 0.8, np.max(dy) * 1.2)
        ax.set_xlabel("Index")
        ax.set_ylabel("Delta Y")
        ax.set_title("Delta Y Position")
        ax.set_xticks(np.arange(len(dy)))
        ax.set_xticklabels([f"{i}" for i in range(len(dy))])

    def render_tilt_y(self, ax, **kwargs):
        tYList = self.tYList
        ax.bar(np.arange(len(tYList)), tYList, **kwargs)
        ax.set_xlabel("Index")
        ax.set_ylabel("Tilt Y")
        ax.set_title("Tilt Y Position")
        ax.set_xticks(np.arange(len(tYList)))
        ax.set_xticklabels([f"{i}" for i in range(len(tYList))])
