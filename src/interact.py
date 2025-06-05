import argparse, time, numpy as np, os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
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

        # ---- load experimental‑setup file into its own namespace ----------
        self.namespace: dict[str, object] = {}
        self._create_axes()
        with open(self.fileName, "r", encoding="utf-8") as f:
            self.expsetup_code = f.read()
            exec(self.expsetup_code, self.namespace)

        self.tunable_vars_setting: dict[str, list] = self.namespace["vars"]
        self.presets: dict[str, dict] | None = self.namespace.get("presets")
        self._plot_type = self.namespace.get("PLOT_TYPE", "Z").upper()
        print(f"Plot type requested by setup: {self._plot_type}")

        # Re‑create main axes in 3‑D if requested
        if self._plot_type == "3D":
            self.ax0.remove()
            self.ax0 = plt.subplot(self.gs[0], projection="3d")
            self.namespace["ax0"] = self.ax0

        # Will be filled by create_sliders() later
        self.sliders: dict[str, Slider | CheckButtons] = {}
        # Keep the fixed parameter order so we can re‑use slot positions
        self.param_order = list(self.tunable_vars_setting.keys())

    # ---------------------------------------------------------------------
    # Figure / axes scaffold
    # ---------------------------------------------------------------------

    def _create_axes(self):
        """Initialise *fig*, *ax0* (main view) and *ax1* side panels."""
        self.fig = plt.figure(figsize=(12, 6))
        self.gs = GridSpec(1, 2, width_ratios=[2.5, 1])
        self.ax0 = plt.subplot(self.gs[0])
        self.gs1 = GridSpecFromSubplotSpec(3, 1, subplot_spec=self.gs[1], hspace=0.3)
        self.ax1 = [plt.subplot(self.gs1[i]) for i in range(3)]
        plt.subplots_adjust(left=0.1, right=0.7)

        # Expose figure & axes so the setup file can draw into them
        self.namespace["fig"] = self.fig
        self.namespace["ax0"] = self.ax0
        self.namespace["gs1"] = self.gs1
        self.namespace["ax1"] = self.ax1

        # Anchor for slider layout (top‑most Y position)
        self.slider_axes_top = 0.95

    # ---------------------------------------------------------------------
    # Internal helpers – clearing, slider <→> param conversion
    # ---------------------------------------------------------------------

    def _clear_axes(self):
        self.ax0.clear()
        for ax in self.ax1:
            ax.clear()

    # ---- slider helpers --------------------------------------------------

    def _get_slider_ax(self, idx: int):
        """Return the *Axes* that hosts the *idx*‑th slider."""
        return self.fig.add_axes(
            [0.8, self.slider_axes_top * (1 - idx / 25), 0.1, 0.03]
        )

    def create_sliders(self):
        """Instantiate all sliders and store them in *self.sliders*."""
        for i, name in enumerate(self.param_order):
            val, vmin, vmax = self.tunable_vars_setting[name]
            ax = self._get_slider_ax(i)
            if vmin is None and vmax is None:
                self.sliders[name] = CheckButtons(ax, [name], [bool(val)])
            else:
                self.sliders[name] = Slider(ax, name, vmin, vmax, valinit=val)

    # ------------------------------------------------------------------
    # Convenience wrappers operating on *self.sliders*
    # ------------------------------------------------------------------

    def _sliders_to_params(self):
        params = {}
        for name, slider in self.sliders.items():
            if isinstance(slider, Slider):
                params[name] = slider.val
            else:  # CheckButtons
                params[name] = slider.get_status()[0]
        return params

    # ------------------------------------------------------------------
    # Core rendering / interaction entry‑points
    # ------------------------------------------------------------------

    def slider_interactive(self):
        """Launch the GUI with sliders and the preset selector."""
        self.create_sliders()

        # --------------------------------------------------------------
        # *update* – redraw main figure when any parameter changes
        # --------------------------------------------------------------
        def _redraw():
            self._clear_axes()
            self.update_table(**self._sliders_to_params())
            if self.render:
                plt.draw()

        def update(_):
            now = time.time()
            if now - self.last_update_time < 1 / self.FPS:
                return
            self.last_update_time = now
            if not self.changing_preset:
                _redraw()

        # Attach callbacks ------------------------------------------------
        for slider in self.sliders.values():
            if isinstance(slider, Slider):
                slider.on_changed(update)
            else:
                slider.on_clicked(update)

        # --------------------------------------------------------------
        # Finetune multiplier block (unchanged)
        # --------------------------------------------------------------
        finetune_ax = self.fig.add_axes([0.78, 0.03, 0.15, 0.12])
        finetune = RadioButtons(finetune_ax, ["x1", "x10", "x100", "x1000"], active=0)
        for side in ("top", "right", "left", "bottom"):
            finetune_ax.spines[side].set_visible(False)
        self.fig.text(0.78, 0.16, "Finetune Multiplier", fontsize=12, weight="bold")

        def _retune(label):
            for idx, name in enumerate(self.param_order):
                slider = self.sliders[name]
                if not isinstance(slider, Slider):
                    continue
                cur = slider.val
                vmin, vmax = self.tunable_vars_setting[name][1:]
                span = vmax - vmin
                match label:
                    case "x1":
                        new_min, new_max = vmin, vmax
                    case "x10":
                        new_min = max(cur - span / 10, vmin)
                        new_max = min(cur + span / 10, vmax)
                    case "x100":
                        new_min = max(cur - span / 100, vmin)
                        new_max = min(cur + span / 100, vmax)
                    case "x1000":
                        new_min = max(cur - span / 1000, vmin)
                        new_max = min(cur + span / 1000, vmax)

                slider.disconnect_events()
                slider.ax.remove()
                new_ax = self._get_slider_ax(idx)
                self.sliders[name] = Slider(new_ax, name, new_min, new_max, valinit=cur)
                self.sliders[name].on_changed(update)
            self.fig.canvas.draw_idle()

        finetune.on_clicked(_retune)

        # --------------------------------------------------------------
        # PRESET selector (RadioButtons)
        # --------------------------------------------------------------
        if self.presets:
            preset_ax = self.fig.add_axes([0.01, 0.8, 0.07, 0.15])
            # Find the current preset
            preset_idx = 0
            for i, (name, preset) in enumerate(self.presets.items()):
                if preset == self.tunable_vars_setting:
                    preset_idx = i
                    break

            preset_rb = RadioButtons(
                preset_ax, list(self.presets.keys()), active=preset_idx
            )
            for side in ("top", "right", "left", "bottom"):
                preset_ax.spines[side].set_visible(False)
            self.fig.text(0.01, 0.95, "Presets", fontsize=12, weight="bold")

            def _load_preset(label):
                preset = self.presets[label]
                self.changing_preset = True
                for name, values in preset.items():
                    if name not in self.sliders:
                        continue
                    target_val, p_min, p_max = values
                    widget = self.sliders[name]

                    if isinstance(widget, Slider):
                        # Re‑create slider if bounds differ
                        if (widget.valmin != p_min) or (widget.valmax != p_max):
                            idx = self.param_order.index(name)
                            widget.disconnect_events()
                            widget.ax.remove()
                            new_ax = self._get_slider_ax(idx)
                            self.sliders[name] = Slider(
                                new_ax, name, p_min, p_max, valinit=target_val
                            )
                            self.sliders[name].on_changed(update)
                        else:
                            widget.set_val(target_val)
                    else:  # CheckButtons (single box)
                        current_state = widget.get_status()[0]
                        desired_state = bool(target_val)
                        if current_state != desired_state:
                            widget.set_active(0)
                self.changing_preset = False
                _redraw()

            preset_rb.on_clicked(_load_preset)

        # Initial draw & show window --------------------------------------
        _redraw()
        plt.show()

        # Dump final values ------------------------------------------------
        print("Final slider values:")
        for k, v in self._sliders_to_params().items():
            print(f"  {k} = {v}")

    # ------------------------------------------------------------------
    # Non‑interactive helpers (logic unchanged)
    # ------------------------------------------------------------------

    def update_table(self, **params):
        self.namespace.update(params)
        exec(self.expsetup_code, self.namespace)

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
            if self.render:
                plt.draw()
                plt.pause(0.01)
            c = float(self.namespace["cost_func"])
            print(f"Iter {n}: cost={c:.5g}; vals={vals}")
            return -c if maximize else c

        result = minimize(cost, initials, bounds=bounds, method="Nelder-Mead")
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
                cost_map[j, i] = float(self.namespace["cost_func"])
        self.render = original_render
        self._clear_axes()
        im = self.ax0.imshow(
            cost_map,
            origin="lower",
            aspect="auto",
            extent=[values_x[0], values_x[-1], values_y[0], values_y[-1]],
            cmap=cmap,
        )
        cbar = self.fig.colorbar(im, ax=self.ax0)
        cbar.set_label("cost_func")
        self.ax0.set_xlabel(param_x)
        self.ax0.set_ylabel(param_y)
        self.ax0.set_title("Parameter scan – cost function map")
        if show:
            plt.show()
        return cost_map


# ---------------------------------------------------------------------
# Small CLI entry‑point ------------------------------------------------
# ---------------------------------------------------------------------
if __name__ == "__main__":
    FILE_NAME = os.path.join(os.path.dirname(__file__), "../demo", "vipa_1st.py")
    MODE = "interact"  # 'interact' | 'optimize' | 'scan'

    table = InteractiveOpticalTable(fileName=FILE_NAME)
    table.render = False  # we'll enable live‑render in the GUI

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
