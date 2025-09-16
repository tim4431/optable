from .optical_component import *
from .component_group import *
from .monitor import *
import matplotlib.pyplot as plt
import numpy as np, time, csv


class OpticalTable:
    def __init__(self):
        self.components = []
        self.rays = []
        self.monitors = []
        self.norender_set = set()

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
        rays = [ray]
        exit_flag = False
        t_start = time.time()
        trace_num = 0
        # MAX_TRACEING_TIME = 0.5
        MAX_TRACEING_TIME = 1e9
        MAX_TRACE_NUM = 500
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
            # print(len(rays))
            # print(rays)
            someone_alive = False
            for ray in rays:
                if ray.alive:
                    someone_alive = True
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
                        rays.remove(ray)
                        rays.extend(rays_min)
                        break
                    # 3. no intersection
                    else:
                        ray.alive = False
            #
            #
            if not someone_alive:
                exit_flag = True
        #
        if not exit_flag:
            print(
                "Ray tracing time exceeds the maximum tracing time after {} traces.".format(
                    trace_num
                )
            )
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

    def gather_rays(self):
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

    def export_rays(self, filename: str):
        """
        Export rays to a file.
        """
        rays_traced = self.gather_rays()

        print(f"Exporting rays to {filename} ...")

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            keys = rays_traced[0].keys() if rays_traced else []
            writer.writerow(keys)
            for ray in rays_traced:
                writer.writerow(ray.values())

    def export_components(
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
