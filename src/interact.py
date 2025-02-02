import argparse, time, numpy as np, os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import minimize


class InteractiveOpticalTable:
    def __init__(self, fileName, FPS=20):
        self.fileName = fileName
        self.last_update_time = time.time()
        self.FPS = FPS
        self.render = True
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
        self.gs = GridSpec(1, 2, width_ratios=[2.5, 1])
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
        # print("Updated table with new parameters.")
        # self.table = self.namespace["table"]

    def _get_slider_ax(self, i):
        return self.fig.add_axes([0.8, 0.95 * (1 - i / 25), 0.1, 0.03])

    def create_sliders(self):
        sliders = {}
        for i, (name, param_range) in enumerate(self.tunable_vars_setting.items()):
            slider_ax = self._get_slider_ax(i)
            val, min_val, max_val = param_range
            if (min_val is None) and (max_val is None):
                # binary variable
                current_val = bool(val)
                sliders[name] = CheckButtons(slider_ax, [name], [current_val])
            else:
                sliders[name] = Slider(
                    slider_ax,
                    name,
                    min_val,
                    max_val,
                    valinit=val,
                )
        return sliders

    def _sliders_to_params(self, sliders):
        # return {name: sliders[name].val for name in self.tunable_vars_setting}
        params = {}
        for name, slider in sliders.items():
            if isinstance(slider, Slider):
                params[name] = slider.val
            elif isinstance(slider, CheckButtons):
                params[name] = slider.get_status()[0]
        return params

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
                # print(params)
                self.update_table(**params)
                if self.render:
                    plt.draw()

        for slider in sliders.values():
            # slider.on_changed(update)
            if isinstance(slider, Slider):
                slider.on_changed(update)
            elif isinstance(slider, CheckButtons):
                slider.on_clicked(update)

        # checkbox_x10_ax = self.fig.add_axes([0.78, 0.05, 0.15, 0.12])
        # finetune_checkbox_x10 = CheckButtons(
        #     checkbox_x10_ax, ["Fine-tune x10"], [False]
        # )
        # checkbox_x100_ax = self.fig.add_axes([0.78, 0.2, 0.15, 0.12])
        # finetune_checkbox_x100 = CheckButtons(
        #     checkbox_x100_ax, ["Fine-tune x100"], [False]
        # )
        finetune_multiplier_ax = self.fig.add_axes([0.78, 0.03, 0.15, 0.12])
        finetune_multiplier = RadioButtons(
            finetune_multiplier_ax,
            ["x1", "x10", "x100"],
            active=0,
            radio_props={"s": [50, 50, 50]},
        )
        finetune_multiplier_ax.spines["top"].set_visible(False)
        finetune_multiplier_ax.spines["right"].set_visible(False)
        finetune_multiplier_ax.spines["left"].set_visible(False)
        finetune_multiplier_ax.spines["bottom"].set_visible(False)
        self.fig.text(0.78, 0.16, "Finetune Multiplier", fontsize=12, weight="bold")

        def toggle_finetune(label):
            for i, (name, param_range) in enumerate(self.tunable_vars_setting.items()):
                # print(name, param_range)
                slider = sliders[name]
                current_val = slider.val
                # print(current_val)
                _, min_val, max_val = param_range
                range_val = max_val - min_val
                if label == "x1":
                    new_valmin = min_val
                    new_valmax = max_val
                elif label == "x10":
                    new_valmin = max(current_val - range_val / 10, min_val)
                    new_valmax = min(current_val + range_val / 10, max_val)
                elif label == "x100":
                    new_valmin = max(current_val - range_val / 100, min_val)
                    new_valmax = min(current_val + range_val / 100, max_val)
                #
                slider.disconnect_events()
                slider.ax.remove()
                slider_newax = self._get_slider_ax(i)
                sliders[name] = Slider(
                    slider_newax,
                    name,
                    new_valmin,
                    new_valmax,
                    valinit=current_val,
                )
                sliders[name].on_changed(update)
                self.fig.canvas.draw_idle()

        finetune_multiplier.on_clicked(toggle_finetune)

        plt.show()

        # print last variables
        params = self._sliders_to_params(sliders)
        print(params)

    def optimize(self, maximize=False):
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
            if self.render:
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
    FILE_NAME = os.path.join(os.path.dirname(__file__), "../demo", "vipa_1st.py")
    MODE = "interact"
    # MODE = "optimize"
    interactive_table = InteractiveOpticalTable(fileName=FILE_NAME)
    interactive_table.render = False

    # Execute based on the chosen mode
    if MODE == "interact":
        interactive_table.slider_interactive()
    elif MODE == "optimize":
        interactive_table.optimize(maximize=True)
