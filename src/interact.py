import argparse, time, numpy as np, os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import minimize


class InteractiveOpticalTable:
    def __init__(self, fileName, FPS=20):
        self.fileName = fileName
        self.last_update_time = time.time()
        self.FPS = FPS
        #
        self.namespace = {}
        self._create_axes()
        with open(self.fileName, "r") as f:
            self.expsetup_code = f.read()
            exec(self.expsetup_code, self.namespace)
        self.tunable_vars_setting = self.namespace["vars"]
        # self._vars = {name: val[0] for name, val in self.tunable_vars_setting.items()}
        #
        self._plot_type = self.namespace.get("PLOT_TYPE", "Z")
        print(f"Plot type: {self._plot_type}")
        if self._plot_type == "3D":
            ax0 = plt.subplot(self.gs[0], projection="3d")
            self.ax0 = ax0
            self.namespace["ax0"] = ax0
        #

    def _create_axes(self):
        self.fig = plt.figure(figsize=(12, 6))
        self.gs = GridSpec(1, 2, width_ratios=[3, 1])
        self.ax0 = plt.subplot(self.gs[0])
        self.gs1 = GridSpecFromSubplotSpec(3, 1, subplot_spec=self.gs[1], hspace=0.3)
        self.ax1 = [plt.subplot(self.gs1[i]) for i in range(3)]
        plt.subplots_adjust(left=0.1, right=0.7)
        #
        self.namespace["fig"] = self.fig
        self.namespace["ax0"] = self.ax0
        self.namespace["gs1"] = self.gs1
        self.namespace["ax1"] = self.ax1
        #

    def _clear_axes(self):
        self.ax0.clear()
        for ax in self.ax1:
            ax.clear()

    def update_table(self, **params):
        """
        Update the optical table configuration with new parameters.
        """
        for key, value in params.items():
            self.namespace[key] = value
        exec(self.expsetup_code, self.namespace)
        self.table = self.namespace["table"]

    def create_sliders(self):
        sliders = {}
        slider_positions = [i / 20 for i in range(len(self.tunable_vars_setting))]
        for i, (name, param_range) in enumerate(self.tunable_vars_setting.items()):
            slider_ax = self.fig.add_axes(
                [0.8, 0.8 * (1 - slider_positions[i]), 0.1, 0.03],
                facecolor="lightgoldenrodyellow",
            )
            val, min_val, max_val = param_range
            sliders[name] = Slider(
                slider_ax,
                name,
                min_val,
                max_val,
                valinit=val,
            )
        return sliders

    def _sliders_to_params(self, sliders):
        return {name: sliders[name].val for name in self.tunable_vars_setting}

    def slider_interactive(self, plot_type="Z"):
        """
        Render the optical table interactively with sliders for each parameter.
        """
        sliders = self.create_sliders()

        def update(val):
            tnow = time.time()
            if tnow - self.last_update_time > 1 / self.FPS:
                self.last_update_time = tnow
                self._clear_axes()

                params = self._sliders_to_params(sliders)
                self.update_table(**params)
                plt.draw()

        for slider in sliders.values():
            slider.on_changed(update)

        plt.show()

        # print last variables
        params = self._sliders_to_params(sliders)
        print(params)

    def optimize(self, render=False, maximize=False):
        """
        Optimize the table configuration to maximize the cost function.
        """
        initial_values = [v[0] for v in self.tunable_vars_setting.values()]
        bounds = [(v[1], v[2]) for v in self.tunable_vars_setting.values()]
        param_names = list(self.tunable_vars_setting.keys())
        #
        n = 0

        def cost_function(vals):
            nonlocal n
            n += 1
            params = {name: val for name, val in zip(param_names, vals)}
            self._clear_axes()
            self.update_table(**params)
            if render:
                plt.draw()
                plt.pause(0.01)
            cost_func = float(self.namespace["cost_func"])
            print("Iteration: ", n, "Cost: ", cost_func, "vals: ", vals)
            return -cost_func if maximize else cost_func

        def _result_to_params(result):
            return {name: result.x[i] for i, name in enumerate(param_names)}

        result = minimize(
            cost_function, initial_values, bounds=bounds, method="Nelder-Mead"
        )
        print("Optimized parameters: ", _result_to_params(result))
        return result


if __name__ == "__main__":
    FILE_NAME = os.path.join(os.path.dirname(__file__), "../demo", "curvemirror.py")
    MODE = "interact"
    # MODE = "optimize"
    interactive_table = InteractiveOpticalTable(fileName=FILE_NAME)

    # Execute based on the chosen mode
    if MODE == "interact":
        interactive_table.slider_interactive()
    elif MODE == "optimize":
        interactive_table.optimize(render=False, maximize=True)
