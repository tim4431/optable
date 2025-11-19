import argparse, time, numpy as np, os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import minimize


class InteractiveOpticalTable:
    """GUI/CLI helper around an experimental setup description file (Python).
    ``vars``
        ``dict`` mapping *parameter name* → ``[value, min, max]``.
        If both ``min`` and ``max`` are ``None`` the parameter is treated as a
        Boolean and rendered as a one‑box ``CheckButtons`` widget.

    Optionally it can expose

    ``presets``
        ``dict[str, dict]`` mapping *preset name* to a *vars‑like* dictionary.
        Each preset entry may list only the parameters it wants to override;
        any parameter omitted from a preset keeps the current slider state.


    ``PLOT_TYPE``
        *Optional*, either ``"Z"`` (default) or ``"3D"``.
    """

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(self, fileName: str, FPS: int = 20):
        self.fileName = fileName
        self.last_update_time = time.time()
        self.FPS = FPS  # target frames per second for slider drags
        self.render = True  # live‑render flag
        self.changing_preset = False  # suppress spurious update() loops
        self.optimization_running = False  # flag for optimization state
        self.optimization_interrupted = False  # flag for optimization interruption

        # ---- load experimental‑setup file into its own namespace ----------
        self.namespace: dict[str, object] = {}
        self._create_axes()
        # — load user experiment file into *self.namespace*
        with open(self.fileName, "r", encoding="utf-8") as f:
            self.expsetup_code = f.read()
            exec(self.expsetup_code, self.namespace)

        self.tunable_vars_setting: dict[str, list] = self.namespace["vars"]
        self.presets: dict[str, dict] | None = self.namespace.get("presets")
        self._plot_type = self.namespace.get("PLOT_TYPE", "Z").upper()
        # print(f"Plot type requested by setup: {self._plot_type}")

        # Re‑create main axes in 3‑D if requested
        if self._plot_type == "3D":
            self.ax0.remove()
            self.ax0 = plt.subplot(self.gs[0], projection="3d")
            self.namespace["ax0"] = self.ax0

        # widgets will be created later
        self.sliders: dict[str, Slider | CheckButtons] = {}
        self.opt_boxes: dict[str, CheckButtons] = {}
        self.param_order = list(self.tunable_vars_setting.keys())

        # detect available cost functions
        self.cost_funcs = self._discover_cost_functions()
        # choose first one by default
        if self.cost_funcs:
            self.selected_cost_name = next(iter(self.cost_funcs))
        self._display_optimization = False

    # ------------------------------------------------------------------ #
    #                    ───  FIGURE / AXES HELPERS  ───                 #
    # ------------------------------------------------------------------ #

    def _create_axes(self):
        """Initialise *fig*, *ax0* (main view) and *ax1* side panels."""
        self.fig = plt.figure(figsize=(12, 6))
        self.gs = GridSpec(1, 2, width_ratios=[2.5, 1])
        self.ax0 = plt.subplot(self.gs[0])
        self.gs1 = GridSpecFromSubplotSpec(3, 1, subplot_spec=self.gs[1], hspace=0.3)
        self.ax1 = [plt.subplot(self.gs1[i]) for i in range(3)]
        plt.subplots_adjust(left=0.1, right=0.7)

        # expose to experiment file
        self.namespace.update(
            dict(fig=self.fig, ax0=self.ax0, gs1=self.gs1, ax1=self.ax1)
        )
        self.slider_axes_top = 0.95

    # ------------------------------------------------------------------ #
    #                              HELPERS                               #
    # ------------------------------------------------------------------ #

    def _clear_axes(self):
        self.ax0.clear()
        for ax in self.ax1:
            ax.clear()

    # ---- slider & checkbox placement helpers -------------------------

    def _get_slider_ax(self, idx: int):
        return self.fig.add_axes(
            [0.8, self.slider_axes_top * (1 - idx / 30), 0.1, 0.03]
        )

    def _get_checkbox_ax(self, idx: int):
        # tiny square to the left of the slider
        return self.fig.add_axes(
            [0.97, self.slider_axes_top * (1 - idx / 30) + 0.005, 0.03, 0.03]
        )

    # ---- cost-function discovery -------------------------------------

    def _discover_cost_functions(self):
        funcs = {}
        for name, val in self.namespace.items():
            if not name.lower().startswith("cost_"):
                continue
            _name = name[5:]  # strip "cost_" prefix
            if callable(val):
                # user supplied a function → call with namespace
                funcs[_name] = lambda ns, f=val: float(f())
            else:
                # numeric variable – wrap in lambda so we always fetch updated value
                funcs[_name] = lambda ns, key=name: float(ns[key])

        return funcs

    # ---- utility conversions -----------------------------------------

    def _sliders_to_params(self):
        params = {}
        for name, slider in self.sliders.items():
            if isinstance(slider, Slider):
                params[name] = slider.val
            else:  # CheckButtons
                params[name] = slider.get_status()[0]
        return params

    def _optimisable_param_names(self):
        """Return list of parameter names whose 'Opt.' box is ticked."""
        return [
            name
            for name, box in self.opt_boxes.items()
            if box.get_status()[0]
            and not (
                self.tunable_vars_setting[name][1] is None
                and self.tunable_vars_setting[name][2] is None
            )
        ]

    # ------------------------------------------------------------------ #
    #                        ───   CORE GUI   ───                        #
    # ------------------------------------------------------------------ #

    def create_sliders(self):
        """Slider + "Opt." checkbox  for every parameter."""
        for i, name in enumerate(self.param_order):
            val, vmin, vmax = self.tunable_vars_setting[name]

            # --- optimisation enable box ---------------------------------
            cb_ax = self._get_checkbox_ax(i)

            # 检查是否为布尔变量(复选框)
            is_boolean_param = vmin is None and vmax is None
            self.opt_boxes[name] = CheckButtons(cb_ax, [""], [False])

            if is_boolean_param:
                # 对于布尔变量，禁用优化选择框
                self.opt_boxes[name].set_active(0)
                cb_ax.set_visible(False)

            for side in ("top", "right", "left", "bottom"):
                cb_ax.spines[side].set_visible(False)

            # --- actual slider / bool toggle ----------------------------
            sl_ax = self._get_slider_ax(i)
            if vmin is None and vmax is None:
                self.sliders[name] = CheckButtons(sl_ax, [name], [bool(val)])
            else:
                self.sliders[name] = Slider(sl_ax, name, vmin, vmax, valinit=val)

    def _update_cost_function_val(self):
        """Update the cost function value display."""
        if hasattr(self, "cost_value_text"):
            cost_value = self._current_cost_value()
            self.cost_value_text.set_text(f"Value: {cost_value:.6g}")

    # ------------------------------------------------------------------ #

    def slider_interactive(self):
        self.create_sliders()

        # ---------- redraw helper --------------------------------------
        def _redraw():
            # print("Redrawing...")
            self._clear_axes()
            self.update_table(**self._sliders_to_params())
            if self.render:
                plt.draw()

        def _on_slider_changed(_):
            if not self.changing_preset:
                now = time.time()
                if now - self.last_update_time < 1 / self.FPS:
                    return
                self.last_update_time = now
                _redraw()

        # attach callbacks
        for w in list(self.sliders.values()) + list(self.opt_boxes.values()):
            if isinstance(w, Slider):
                w.on_changed(_on_slider_changed)
            else:
                w.on_clicked(_on_slider_changed)

        # BLOCKS
        # --------------------------------------------------------------
        # Finetune
        # --------------------------------------------------------------
        FINETUNE_X, FINETUNE_Y = 0.02, 0.01
        FINTUNE_DX, FINTUNE_DY = 0.15, 0.11
        finetune_ax = self.fig.add_axes(
            [FINETUNE_X, FINETUNE_Y, FINTUNE_DX, FINTUNE_DY]
        )
        finetune = RadioButtons(finetune_ax, ["x1", "x10", "x100", "x1000"], active=0)
        for side in ("top", "right", "left", "bottom"):
            finetune_ax.spines[side].set_visible(False)
        self.fig.text(
            FINETUNE_X,
            FINETUNE_Y + 0.11,
            "Finetune Multiplier",
            fontsize=12,
            weight="bold",
        )

        def _retune(label):
            for idx, name in enumerate(self.param_order):
                slider = self.sliders[name]
                if not isinstance(slider, Slider):
                    continue
                cur = slider.val
                vmin, vmax = self.tunable_vars_setting[name][1:]
                span = vmax - vmin
                factor = dict(x1=1, x10=10, x100=100, x1000=1000)[label]
                new_min = max(cur - span / factor, vmin)
                new_max = min(cur + span / factor, vmax)

                slider.disconnect_events()
                slider.ax.remove()
                new_ax = self._get_slider_ax(idx)
                self.sliders[name] = Slider(new_ax, name, new_min, new_max, valinit=cur)
                self.sliders[name].on_changed(_on_slider_changed)
            self.fig.canvas.draw_idle()

        finetune.on_clicked(_retune)

        # --------------------------------------------------------------
        # Presets
        # --------------------------------------------------------------
        if self.presets:
            preset_ax = self.fig.add_axes([0.01, 0.8, 0.1, 0.15])
            for side in ("top", "right", "left", "bottom"):
                preset_ax.spines[side].set_visible(False)
            # Find the current preset

            preset_idx = 0
            for i, (name, preset) in enumerate(self.presets.items()):
                if preset == self.tunable_vars_setting:
                    preset_idx = i
                    break

            preset_rb = RadioButtons(
                preset_ax, list(self.presets.keys()), active=preset_idx
            )
            self.fig.text(0.02, 0.95, "Presets", fontsize=12, weight="bold")

            def _load_preset(label):
                preset = self.presets[label]
                self.changing_preset = True
                for name, values in preset.items():
                    if name not in self.sliders:
                        continue
                    target_val, pmin, pmax = values
                    widget = self.sliders[name]

                    if isinstance(widget, Slider):
                        # Re‑create slider if bounds differ
                        if (widget.valmin != pmin) or (widget.valmax != pmax):
                            idx = self.param_order.index(name)
                            widget.disconnect_events()
                            widget.ax.remove()
                            new_ax = self._get_slider_ax(idx)
                            self.sliders[name] = Slider(
                                new_ax, name, pmin, pmax, valinit=target_val
                            )
                            self.sliders[name].on_changed(_on_slider_changed)
                        else:
                            widget.set_val(target_val)
                    else:  # CheckButtons (single box)
                        current_state = widget.get_status()[0]
                        desired_state = bool(target_val)
                        if current_state != desired_state:
                            widget.set_active(0)
                self.changing_preset = False
                _on_slider_changed(None)

            preset_rb.on_clicked(_load_preset)

        # --------------------------------------------------------------
        # Cost function selector + OPTIMISE button
        # --------------------------------------------------------------
        if self._display_optimization:
            COST_X, COST_Y = 0.03, 0.38
            cost_rb_ax = self.fig.add_axes([COST_X, COST_Y, 0.17, 0.1])
            self.fig.text(
                COST_X, COST_Y + 0.12, "Cost Function", fontsize=12, weight="bold"
            )
            BTN_X, BTN_Y = COST_X + 0.12, COST_Y + 0.10
            btn_ax = self.fig.add_axes([BTN_X, BTN_Y, 0.06, 0.04])
            btn = Button(btn_ax, "OPTIMISE", hovercolor="lightgray")
            for s in cost_rb_ax.spines.values():
                s.set_visible(False)

            def _pick_cost(label):
                self.selected_cost_name = label
                self._update_cost_function_val
                self.fig.canvas.draw_idle()

            # # add radio buttons for cost functions
            # if self.cost_funcs:
            #     cost_rb = RadioButtons(
            #         cost_rb_ax, list(self.cost_funcs.keys()), active=0
            #     )
            #     cost_rb.on_clicked(_pick_cost)
            # add current cost value display
            COST_VAL_X, COST_VAL_Y = COST_X, COST_Y + 0.10
            cost_value_text_ax = self.fig.add_axes([COST_VAL_X, COST_VAL_Y, 0.17, 0.02])
            cost_value_text_ax.axis("off")
            self.cost_value_text = cost_value_text_ax.text(
                0, 0, "Value: --", fontsize=10
            )

            #
            COST_PROGRESS_X, COST_PROGRESS_Y = COST_X, COST_Y - 0.18
            self.opt_progress_ax = self.fig.add_axes(
                [COST_PROGRESS_X, COST_PROGRESS_Y, 0.17, 0.15]
            )
            self.opt_progress_ax.set_xlabel("Iter Num", fontsize=8)
            self.opt_progress_ax.set_ylabel("Cost Function", fontsize=8)
            self.opt_progress_ax.set_yscale("log")
            (self.opt_progress_line,) = self.opt_progress_ax.plot([], [], "b-")
            self.opt_progress_ax.grid(True)

            def _on_slider_changed_with_cost_update(_):
                if not self.changing_preset:
                    now = time.time()
                    if now - self.last_update_time < 1 / self.FPS:
                        return
                    self.last_update_time = now
                    _redraw()
                    # 更新代价函数值显示
                    if hasattr(self, "cost_value_text"):
                        cost_value = self._current_cost_value()
                        self.cost_value_text.set_text(f"Value: {cost_value:.6g}")

            # 替换之前的回调函数
            for w in list(self.sliders.values()) + list(self.opt_boxes.values()):
                if isinstance(w, Slider):
                    w.on_changed(_on_slider_changed_with_cost_update)
                else:
                    w.on_clicked(_on_slider_changed_with_cost_update)

            if self.cost_funcs:
                cost_rb = RadioButtons(
                    cost_rb_ax, list(self.cost_funcs.keys()), active=0
                )
                cost_rb.on_clicked(_pick_cost)
                cost_value = self._current_cost_value()
                self.cost_value_text.set_text(f"Value: {cost_value:.6g}")

            # 修改按钮点击处理函数
            def _run(_):
                # click to stop optimization if running
                if self.optimization_running:
                    self.optimization_interrupted = True
                else:  # click to start optimization
                    self.optimization_running = True
                    self.optimization_interrupted = False

                    original_color = btn.color
                    original_hovercolor = btn.hovercolor
                    original_label = btn.label.get_text()

                    btn.color = "lightgreen"
                    btn.hovercolor = "red"
                    btn.label.set_text("Optimizing...")
                    btn._hover_fill_color = "red"  # 确保悬停颜色更新

                    # 添加悬停时的tooltip文本
                    orig_tooltip = btn.ax.get_title()
                    btn.ax.set_title("Click to abort")
                    self.fig.canvas.draw_idle()

                    plt.draw()
                    plt.pause(0.01)

                    try:
                        self.optimize_selected()
                    finally:
                        # 恢复按钮原始状态
                        self.optimization_running = False
                        btn.color = original_color
                        btn.hovercolor = original_hovercolor
                        btn.label.set_text(original_label)
                        btn._hover_fill_color = original_hovercolor
                        btn.ax.set_title(orig_tooltip)
                        self.fig.canvas.draw_idle()

            btn.on_clicked(_run)

        # --------------------------------------------------------------
        # END of blocks
        # --------------------------------------------------------------
        _redraw()
        plt.show()

        # Dump final values ------------------------------------------------
        print("Final slider values:")
        for k, v in self._sliders_to_params().items():
            print(f"  {k} = {v}")

        # Print final slider values for easy copy, for example:
        # sol0 = {
        # "V2dX": [0.26746, -2, 2],
        # "V2dY": [0, -0.02, 0.7],
        # "V2dXMLA": [3.30596, 1, 15],
        # "V2ADD_MLA": [1, None, None], # for Boolean parameters, use None for min/max

        print("\nFinal parameters for copy-paste:")
        print("sol0 = {")
        for name, slider in self.sliders.items():
            if isinstance(slider, Slider):
                val = slider.val
                vmin, vmax = slider.valmin, slider.valmax
            else:
                val = slider.get_status()[0]
                vmin, vmax = None, None
            print(f'    "{name}": [{val}, {vmin}, {vmax}],')
        print("}")

    #                       ───  BACK-END LOGIC  ───                      #
    # ------------------------------------------------------------------ #

    def update_table(self, **params):
        self.namespace.update(params)
        exec(self.expsetup_code, self.namespace)

    # ------------ optimisation (selected subset) -----------------------

    def _current_cost_value(self):
        val = self.cost_funcs[self.selected_cost_name](self.namespace)
        return float(val)

    def optimize_selected(self, maximise=False):
        opt_names = self._optimisable_param_names()
        if not opt_names:
            print("No parameter selected for optimisation.")
            return

        # separate vectors for varying vs fixed parameters
        fixed = {}
        initials, bounds = [], []
        for n in self.param_order:
            w = self.sliders[n]
            if n in opt_names:
                initials.append(w.val)
                if isinstance(w, Slider):
                    bounds.append((w.valmin, w.valmax))
            else:
                fixed[n] = w.val if isinstance(w, Slider) else w.get_status()[0]

        # initialise optimisation progress plot
        iterations = []
        costs = []
        self.opt_progress_line.set_data(iterations, costs)
        # logplot in y axis
        # self.opt_progress_ax.set_yscale("log")
        self.opt_progress_ax.relim()
        self.opt_progress_ax.autoscale_view()

        iteration_count = [0]

        # -------------------------------------------------------------- #
        def _cost(x):
            if self.optimization_interrupted:
                raise InterruptedError("Optimization aborted by user")

            iteration_count[0] += 1
            params = fixed | {n: v for n, v in zip(opt_names, x)}
            self._clear_axes()
            self.update_table(**params)

            c = self._current_cost_value()
            print(f"Iter {iteration_count[0]}: cost={c: .6g}")
            iterations.append(iteration_count[0])
            costs.append(c if not maximise else -c)

            # update main drawing every 10 iterations
            if iteration_count[0] % 10 == 0 and self.render:
                self._update_cost_function_val()
                self.opt_progress_line.set_data(iterations, costs)
                self.opt_progress_ax.relim()
                self.opt_progress_ax.autoscale_view()
                self.fig.canvas.draw_idle()
                plt.draw()
                plt.pause(0.01)

            return -c if maximise else c

        print(f"Optimising over: {opt_names}")
        # plt.ion()
        try:
            res = minimize(
                _cost,
                initials,
                bounds=bounds,
                method="Nelder-Mead",
                options={
                    "maxiter": 150,  # Larger max number of iterations
                    "maxfev": 150,  # Larger max number of function evaluations
                    "xatol": 1e-5,  # Smaller tolerance on parameter change
                    "fatol": 1e-7,  # Smaller tolerance on function value change
                    # "adaptive": True,  # Enable adaptive step size (optional, good for badly scaled problems)
                },
            )
        except InterruptedError:
            print("Optimization interrupted by user.")
            res = None

        # 清除中断标志
        self.optimization_interrupted = False

        if res is not None:
            # plt.ioff()
            print("Result:")
            for n, v in zip(opt_names, res.x):
                print(f"  {n} = {v}")
            # push final values back to sliders for visual feedback
            for n, v in zip(opt_names, res.x):
                w = self.sliders[n]
                if isinstance(w, Slider):
                    w.set_val(v)
                else:
                    desired = bool(round(v))
                    if w.get_status()[0] != desired:
                        w.set_active(0)

        # update the cost function value display
        self._update_cost_function_val()

        # one last redraw with finished state
        self._clear_axes()
        self.update_table(**self._sliders_to_params())
        plt.draw()

    def optimize(self, maximize=False):
        initials = [v[0] for v in self.tunable_vars_setting.values()]
        bounds = [(v[1], v[2]) for v in self.tunable_vars_setting.values()]
        names = self.param_order
        n = 0

        def cost(vals):
            nonlocal n
            n += 1
            params = {name: val for name, val in zip(names, vals)}
            self._clear_axes()
            self.update_table(**params)
            c = float(self.namespace["cost_func"])
            print(f"Iter {n}: cost={c:.5g}; vals={vals}")
            return -c if maximize else c

        result = minimize(cost, initials, bounds=bounds, method="Nelder-Mead", tol=1e-6)
        print("Optimized parameters:")
        for name, val in zip(names, result.x):
            print(f"  {name} = {val}")
        return result

    def scan(
        self,
        param_x: str,
        values_x,
        param_y: str,
        values_y,
        *,
        cmap="viridis",
        show=True,
    ):
        Nx, Ny = len(values_x), len(values_y)
        cost_map = np.empty((Ny, Nx), dtype=float)
        original_render = self.render
        self.render = False
        for j, y in enumerate(values_y):
            for i, x in enumerate(values_x):
                self.update_table(**{param_x: x, param_y: y})
                cost_map[j, i] = self._current_cost_value()
        self.render = original_render
        self._clear_axes()
        im = self.ax0.imshow(
            cost_map,
            origin="lower",
            aspect="auto",
            extent=[values_x[0], values_x[-1], values_y[0], values_y[-1]],
            cmap=cmap,
        )
        self.fig.colorbar(im, ax=self.ax0).set_label(self.selected_cost_name)
        self.ax0.set_xlabel(param_x)
        self.ax0.set_ylabel(param_y)
        self.ax0.set_title("Parameter scan – cost map")
        if show:
            plt.show()
        return cost_map


# ---------------------------------------------------------------------
# Small CLI entry‑point ------------------------------------------------
# ---------------------------------------------------------------------
if __name__ == "__main__":
    FILE_NAME = os.path.join(
        os.path.dirname(__file__),
        "../demo",
        # "1d_beam_focusing.py",
        "ripa_gen2_2nd.py",
        # "fig1a_wavefront.py",
        # "2d_beam_array_slm.py",
        # os.path.dirname(__file__),
        # "../examples",
        # "corner_cube.py",
    )
    MODE = "interact"  # 'interact' | 'optimize' | 'scan'

    table = InteractiveOpticalTable(fileName=FILE_NAME)
    table._display_optimization = False  # enable cost function display
    # table._display_optimization = True  # enable cost function display

    match MODE:
        case "interact":
            table.slider_interactive()
        case "optimize":
            table.optimize(maximize=True)
        case "scan":
            table.scan(
                "V2d4F",
                np.linspace(-3.3, 2.7, 11),
                "V2dXMLA",
                np.linspace(0.7, 6.7, 11),
            )
