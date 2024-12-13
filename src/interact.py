from optical_table import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import numpy as np
import os, time


class InteractiveOpticalTable:
    def __init__(self, fileName):
        self.last_update_time = time.time()
        self.FPS = 20
        self.renderpanel = False
        self.ax0 = None
        self.namespace = {}

        #
        # # Set up the figure and axis
        # if (not self.renderpanel) or (len(self.table.monitors) == 0):
        #     fig, ax0 = plt.subplots(1, 1, figsize=(12, 6))
        # else:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
        gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.3)
        ax1 = [plt.subplot(gs1[i]) for i in range(3)]

        self.fig = fig
        self.ax0 = ax0
        self.gs1 = gs1
        self.ax1 = ax1
        #
        self.namespace["fig"] = fig
        self.namespace["ax0"] = ax0
        self.namespace["gs1"] = gs1
        self.namespace["ax1"] = ax1
        plt.subplots_adjust(left=0.1, right=0.7)
        #
        # Initial execution of expsetup.py code
        with open(fileName, "r") as f:
            self.expsetup_code = f.read()
        exec(self.expsetup_code, self.namespace)
        self.tunable_params = self.namespace["vars"]
        self.table = self.namespace["table"]
        #
        self._plot_type = self.namespace.get("PLOT_TYPE", "Z")
        print(f"Plot type: {self._plot_type}")
        if self._plot_type == "3D":
            ax0 = plt.subplot(gs[0], projection="3d")
            self.ax0 = ax0
            #
            self.namespace["ax0"] = ax0

    def update_table(self, **params):
        """
        Update the optical table configuration based on the slider values.
        """
        for key, value in params.items():
            self.namespace[key] = value

        exec(self.expsetup_code, self.namespace)
        self.table = self.namespace["table"]

    def render_interactive(self, type: str = "Z"):
        """
        Render the optical table interactively using sliders to adjust parameters.
        """
        # Create sliders for each tunable parameter
        num_params = len(self.tunable_params)

        # Get initial parameter values from the namespace
        initial_params = {name: self.namespace[name] for name in self.tunable_params}
        self.update_table(**initial_params)
        # self.table.render(self.ax0, type=type)

        sliders = {}
        slider_positions = [i / 20 for i in range(num_params)]
        for i, name in enumerate(self.tunable_params):
            slider_ax = self.fig.add_axes(
                [0.8, 0.8 * (1 - slider_positions[i]), 0.1, 0.03],
                facecolor="lightgoldenrodyellow",
            )
            param_slider = self.tunable_params[name]
            val = param_slider[0]
            start = param_slider[1]
            end = param_slider[2]
            sliders[name] = Slider(
                slider_ax,
                name,
                start,
                end,
                valinit=val,
                # orientation="vertical",
            )

        def update(val):
            tnow = time.time()
            if tnow - self.last_update_time > 1 / self.FPS:
                self.last_update_time = tnow
                self.ax0.clear()
                for ax in self.ax1:
                    ax.clear()
                #
                params = {name: sliders[name].val for name in self.tunable_params}
                self.update_table(**params)
                #
                plt.draw()

        for slider in sliders.values():
            slider.on_changed(update)

        plt.show()

        # print current variables
        params = {name: sliders[name].val for name in self.tunable_params}
        print(params)


interactive_table = InteractiveOpticalTable(fileName="../demo/test.py")
interactive_table.render_interactive()
