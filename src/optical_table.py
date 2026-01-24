from .optical_component import *
from .component_group import *
from .monitor import *
import matplotlib.pyplot as plt
import numpy as np, time, csv


class OpticalTable:
    def __init__(self, **kwargs):
        self.components = []
        self.rays = []
        self.monitors = []
        self.norender_set = set()
        self._bbox = (
            None,
            None,
            None,
            None,
            None,
            None,
        )  # (xmin, xmax, ymin, ymax, zmin, zmax)
        self.unit = kwargs.get("unit", 1e-2)  # default unit is cm

    def add_components(self, component: Union[OpticalComponent, List]):
        if isinstance(component, OpticalComponent):
            self.components.append(component)
        elif isinstance(component, list):
            for c in component:
                if isinstance(c, OpticalComponent):
                    self.components.append(c)
                elif isinstance(c, list):
                    self.components.extend(c)

    def add_monitors(self, monitor: Union[Monitor, List]):
        if isinstance(monitor, Monitor):
            self.monitors.append(monitor)
        elif isinstance(monitor, list):
            for m in monitor:
                if isinstance(m, Monitor):
                    self.monitors.append(m)
                elif isinstance(m, list):
                    self.monitors.extend(m)

    @property
    def bbox(self):
        if self._bbox[0] is None:
            self.get_bbox()
        return self._bbox

    def get_bbox(self):
        bboxes = [c.bbox for c in self.components]
        bbox = base_merge_bboxs(bboxes)
        self._bbox = bbox
        return bbox

    def ray_tracing(self, rays: Union[Ray, List[Ray]], perfomance_limit=None):
        """
        Perform ray tracing simulation.

        Parameters:
            rays: Ray or list of Ray objects.
        """
        if isinstance(rays, Ray):
            rays = [rays]
        for ray in rays:
            rays_traced = self._single_ray_tracing(
                ray, perfomance_limit=perfomance_limit
            )
            self.rays.extend(rays_traced)
        #
        return copy.deepcopy(self.rays)

    def _single_ray_tracing(self, ray: Ray, perfomance_limit=None):
        """
        Perform ray tracing simulation.
        """
        alive_rays = [ray]
        dead_rays = []
        exit_flag = False
        t_start = time.time()
        last_hint_time = t_start
        trace_num = 0
        # MAX_TRACEING_TIME = 0.5
        MIN_HINTING_TIME = 1
        MAX_TRACEING_TIME = 600
        MAX_TRACE_NUM = 2000
        if perfomance_limit is not None:
            if "max_trace_time" in perfomance_limit:
                MAX_TRACEING_TIME = perfomance_limit["max_trace_time"]
            if "max_trace_num" in perfomance_limit:
                MAX_TRACE_NUM = perfomance_limit["max_trace_num"]
        while (
            (not exit_flag)
            and (time.time() - t_start < MAX_TRACEING_TIME)
            and (trace_num < MAX_TRACE_NUM)
        ):
            trace_num += 1
            # print every second, how much alive traces left, how many traces have been done
            tnow = time.time()
            if (tnow - t_start > MIN_HINTING_TIME) and (
                tnow - last_hint_time > MIN_HINTING_TIME
            ):
                num_alive = len(alive_rays)
                num_dead = len(dead_rays)
                print(
                    "Tracing... Time elapsed: {:.2f} s, Trace num: {}, Alive rays: {}, Dead rays: {}".format(
                        tnow - t_start, trace_num, num_alive, num_dead
                    )
                )
                last_hint_time = tnow
            # print(len(rays))
            # print(rays)
            if len(alive_rays) > 0:
                ray = alive_rays.pop(0)
                # 1. find the component that the ray will intersect the first
                t_min = None
                rays_min = None
                for component in self.components:
                    t, new_rays = component.interact(ray)
                    if t is not None and (t_min is None or t < t_min):
                        t_min = t
                        rays_min = new_rays
                # 2. intersection
                if rays_min is not None:
                    for r in rays_min:
                        if r.alive:
                            alive_rays.append(r)
                        else:
                            dead_rays.append(r)
                # 3. no intersection
                else:
                    # ray.alive = False
                    dead_rays.append(ray)
            else:
                exit_flag = True
        #
        if not exit_flag:
            print(
                "Ray tracing time exceeds the maximum tracing time after {} traces.".format(
                    trace_num
                )
            )
        rays = dead_rays.copy()
        for monitor in self.monitors:
            monitor.record(rays)
        return rays

    def render(self, ax=None, type: str = "Z", roi=None, **kwargs):
        if type == "Z":
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.subplots_adjust(left=0.1, right=0.7)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            # print(self.norender_set)
            for ray in self.rays:
                # print(ray._id)
                if ray._id not in self.norender_set:
                    ray.render(ax, type="Z", **kwargs)
            for component in self.components:
                if component._id not in self.norender_set:
                    component.render(ax, type="Z", **kwargs)
            for monitor in self.monitors:
                if monitor._id not in self.norender_set:
                    monitor.render(ax, type="Z", **kwargs)
            # ax.set_aspect("equal")
            ax.set_aspect("auto")
        elif type == "3D":
            if ax is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            for ray in self.rays:
                if ray._id not in self.norender_set:
                    ray.render(ax, type="3D", **kwargs)
            for component in self.components:
                if component._id not in self.norender_set:
                    component.render(ax, type="3D", **kwargs)
            for monitor in self.monitors:
                if monitor._id not in self.norender_set:
                    monitor.render(ax, type="3D", **kwargs)
            ax.set_aspect("equal")
        else:
            raise ValueError(f"render: Invalid type: {type}")
        #
        if roi is not None:
            ax.set_xlim(roi[0], roi[1])
            ax.set_ylim(roi[2], roi[3])
            if ax.name == "3d":
                ax.set_zlim(roi[4], roi[5])
            aspect = kwargs.get("aspect", "equal")
            ax.set_aspect(aspect)

    # >>> ABCD MATRIX CALCULATION
    def calculate_abcd_matrix(
        self,
        mon0: Monitor,
        mon1: Monitor,
        rays: List[Ray],
        disp=1e-5,
        rot=1e-5,
        debugaxs=None,
    ) -> np.ndarray:
        """Calculate the ABCD matrix between two monitors mon0 and mon1 (mon0 to mon1).
        default principal axis is "Y".
        raysids are the ids of the rays used to calculate the matrix. If None, use all rays that intersect with mon0.
        The principal axis for each monitor is specified by pax0 and pax1, it can be either "Y" or "Z" or a custom vector lying on the plane of the monitor.
        delta is the small perturbation applied to calculate the matrix elements.

        Returns:
            A Nray x 2 x 2 numpy array representing the ABCD matrix for each ray.
        """
        #
        pax0_dispvec = mon0.tangent_Y
        pax0_rotvec = mon0.tangent_Z

        def _simulate(rays):
            self.rays = []
            mon0.clear()
            mon1.clear()
            self.ray_tracing(rays)
            if debugaxs is not None:
                self.render(ax=debugaxs, type="Z")

        #
        # VALIDATIONS
        assert len(rays) > 0, "No rays to trace in ABCD calculation."
        Nrays = len(rays)
        raysids = [ray._id for ray in rays]
        assert len(set(raysids)) == Nrays, "Redundant ray ids in ABCD calculation."
        # sort rays by their ids
        rays_idx_sortbyids = np.argsort(raysids)
        rays = [rays[i] for i in rays_idx_sortbyids]
        #
        # RAY TRACING
        _simulate(rays)
        # get rays intersecting with monitors
        rays_mon0 = mon0.get_rays(sort="ID")
        raysid_mon0 = [ray._id for ray in rays_mon0]
        assert set(raysids) == set(
            raysid_mon0
        ), "Rays at mon0 do not match the input rays."
        PListmon0 = mon0.get_PList(sort="ID")
        rays_mon1 = mon1.get_rays(sort="ID")
        raysid_mon1 = [ray._id for ray in rays_mon1]
        assert set(raysids) == set(
            raysid_mon1
        ), "Rays at mon1 do not match the input rays."
        yList00 = mon1.get_yList(sort="ID")
        tYList00 = mon1.get_tYList(sort="ID")
        #
        Ms = np.zeros((Nrays, 2, 2))  # store each ray's ABCD matrix
        # now bias rays in principal axis, and perform ray tracing
        rays_biased = []
        # first bias in position
        for idx in range(Nrays):
            rays_biased.append(rays[idx]._Translate(pax0_dispvec * disp))
        _simulate(rays_biased)
        yList10 = mon1.get_yList(sort="ID")
        tYList10 = mon1.get_tYList(sort="ID")
        A = (yList10 - yList00) / disp
        C = (tYList10 - tYList00) / disp
        # then bias in angle
        rays_biased = []
        for idx in range(Nrays):
            rays_biased.append(
                rays[idx]._RotAround(pax0_rotvec, PListmon0[idx], rot)
            )  # rotate around the intersection point on mon0
        _simulate(rays_biased)
        yList01 = mon1.get_yList(sort="ID")
        tYList01 = mon1.get_tYList(sort="ID")
        B = (yList01 - yList00) / rot
        D = (tYList01 - tYList00) / rot
        #
        Ms[:, 0, 0] = A
        Ms[:, 0, 1] = B
        Ms[:, 1, 0] = C
        Ms[:, 1, 1] = D

        return Ms

    @staticmethod
    def calibrate_symmetric_4f(
        lens: Union[OpticalComponent, ComponentGroup],
        rays: List[Ray],
        F10: float,
        F20: float,
        criterion: str = "M=-I",
        debugaxs=None,
        optimize=True,
        display_M=False,
    ):
        """Calibrate a symmetric 4f system formed by two lenses and two monitors.
        The distances are mon0 - F1 - lens - 2*F2 - lens(rotated 180) - F1 - mon1.
        The function will adjust the distances F1 and F2 to achieve the desired 4f imaging condition.

        Parameters:
            lens: OpticalComponent, the lens to be calibrated.
            rays: List[Ray], the rays to be traced.
            F10,F20: float, the initial distances.
            debugaxs: matplotlib axes, optional, for debugging visualization.
        Returns:
            F1, F2: calibrated distances.
        """
        from scipy.optimize import minimize
        from tqdm import tqdm

        # create a copy of lens
        def simulate(F1, F2):
            dr0 = np.array([F1, 0, 0]) - lens.origin
            l0 = lens.copy()._Translate(dr0)
            dr1 = np.array([F1 + 2 * F2, 0, 0]) - lens.origin
            l1 = lens.copy()._Translate(dr1).RotZ(np.pi)
            Mon0 = Monitor(origin=[0, 0, 0], width=5, height=5)
            Mon1 = Monitor(origin=[2 * F1 + 2 * F2, 0, 0], width=5, height=5)
            table = OpticalTable()
            table.add_components([l0, l1])
            table.add_monitors([Mon0, Mon1])
            Ms = table.calculate_abcd_matrix(Mon0, Mon1, rays)
            #
            if not optimize:
                table.ray_tracing(rays)
                print(
                    "Rendering 4f system with F1={:.4f}, F2={:.4f} ...".format(F1, F2)
                )
                table.render(ax=debugaxs, type="Z" if debugaxs is not None else None)
            return Ms

        def _cost_func_M_equal_mI(F1, F2):
            Ms = simulate(F1, F2)
            cost = 0
            for M in Ms:
                cost += np.linalg.norm(M + np.eye(2))
            cost /= len(Ms)
            if display_M:
                for M in Ms:
                    print(M)
            pbar.set_description(f"Testing F1={F1:.4f}, F2={F2:.4f}, cost={cost:.6f}")
            return cost

        def _cost_func_flat_field(F1, F2):
            Ms = simulate(F1, F2)
            cost = 0
            for M in Ms:
                A = M[0, 0]
                B = M[0, 1]
                C = M[1, 0]
                D = M[1, 1]
                d0 = 1.5
                ds = (d0 * A - B) / (D - C * d0)
                cost += np.abs(ds - d0)
            cost /= len(Ms)
            pbar.set_description(f"Testing F1={F1:.4f}, F2={F2:.4f}, cost={cost:.6f}")
            return cost

        def cost_function():
            # if "pbar" in locals():
            pbar.update(1)
            if criterion == "M=-I":
                return _cost_func_M_equal_mI
            elif criterion == "flat_field":
                return _cost_func_flat_field
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

        if not optimize:
            simulate(F10, F20)
            return F10, F20
        else:
            pbar = tqdm(total=500, desc="Optimizing 4f system")
            res = minimize(
                lambda x: cost_function()(x[0], x[1]),
                x0=[F10, F20],
                method="Nelder-Mead",
                options={"disp": True, "xatol": 1e-5, "maxiter": 50},
            )
            pbar.close()
            F1_opt, F2_opt = res.x
            return F1_opt, F2_opt

    # >>> VISUALIZATION FUNCTIONS
    def add_wavelength_legend(self, ax, wavelengths):
        """
        Adds a custom color legend to an existing ax for specified wavelengths.
        wavelengths: list of floats/ints in nm
        """
        from matplotlib.lines import Line2D

        # Create the 'Proxy' line objects for the legend
        # We use the wavelength_to_rgb function defined earlier
        proxies = []
        for wl in wavelengths:
            wl_m = wl * self.unit  # convert to nm
            color = wavelength_to_rgb(wl_m)
            line = Line2D([0], [0], color=color, lw=3, label=f"{int(wl_m*1e9)} nm")
            proxies.append(line)

        # Optional: Merge with existing legend items if the plot already has them
        existing_handles, _ = ax.get_legend_handles_labels()

        # Update the legend
        ax.legend(handles=existing_handles + proxies, loc="best")

    # >>> EXPORTING FUNCTIONS
    def gather_rays_csv(self):
        """
        Gather all rays in the optical table.
        """

        def _repr_self_dict(ray):
            d = {
                "origin": to_mathematical_str(str(ray.origin.tolist())),
                "transform_matrix": to_mathematical_str(
                    str(ray.transform_matrix.tolist())
                ),
                "intensity": get_attr_str(ray, "intensity", "None"),
                "length": get_attr_str(ray, "length", "None"),
                "qo": to_mathematical_str(str(get_attr_str(ray, "qo", "None"))),
                "n": to_mathematical_str(str(get_attr_str(ray, "n", "None"))),
            }
            return d

        rays = []
        for ray in self.rays:
            rays.append(_repr_self_dict(ray))
        return rays

    def gather_components(
        self, avoid_flatten_classname: List = [], ignore_classname: List = []
    ) -> List[dict]:
        """
        Gather all components in the optical table, avoiding flattening if specified.
        """
        components = []
        for component in self.components:
            components.extend(
                component.gather_components(
                    avoid_flatten_classname=avoid_flatten_classname,
                    ignore_classname=ignore_classname,
                )
            )
        return components

    def export_rays_csv(self, filename: str):
        """
        Export rays to a file.
        """
        rays_traced = self.gather_rays_csv()

        print(f"Exporting rays to {filename} ...")

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            keys = rays_traced[0].keys() if rays_traced else []
            writer.writerow(keys)
            for ray in rays_traced:
                writer.writerow(ray.values())

    def export_components_csv(
        self,
        filename: str,
        avoid_flatten_classname: List = [],
        ignore_classname: List = [],
    ):
        """
        Export components to a file. Including its class, origin and normal vector
        """
        components = self.gather_components(
            avoid_flatten_classname=avoid_flatten_classname,
            ignore_classname=ignore_classname,
        )

        print(f"Exporting components to {filename} ...")

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            keys = components[0].keys() if components else []
            writer.writerow(keys)
            for component in components:
                writer.writerow(component.values())
