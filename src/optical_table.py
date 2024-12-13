from optical_components import *

import matplotlib.pyplot as plt
import numpy as np, time


class OpticalTable:
    def __init__(self):
        self.components = []
        self.rays = []
        self.monitors = []

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

    def ray_tracing(self, rays: Union[Ray, List[Ray]]):
        """
        Perform ray tracing simulation.

        Parameters:
            rays: Ray or list of Ray objects.
        """
        if isinstance(rays, Ray):
            rays = [rays]
        for ray in rays:
            rays_traced = self._single_ray_tracing(ray)
            self.rays.extend(rays_traced)
        #
        return copy.deepcopy(self.rays)

    def _single_ray_tracing(self, ray: Ray):
        """
        Perform ray tracing simulation.
        """
        rays = [ray]
        exit_flag = False
        t_start = time.time()
        trace_num = 0
        MAX_TRACEING_TIME = 0.1
        MAX_TRACE_NUM = 500
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
            for ray in self.rays:
                ray.render(ax, type="Z", **kwargs)
            for component in self.components:
                component.render(ax, type="Z", **kwargs)
            for monitor in self.monitors:
                monitor.render(ax, type="Z", **kwargs)
            ax.set_aspect("equal")
        elif type == "3D":
            if ax is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            for ray in self.rays:
                ray.render(ax, type="3D", **kwargs)
            for component in self.components:
                component.render(ax, type="3D", **kwargs)
            for monitor in self.monitors:
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
            ax.set_aspect("equal")
