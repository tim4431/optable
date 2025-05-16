import argparse, time, numpy as np, os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import minimize


class InteractiveOpticalTable:
    """GUI/CLI helper around an experimental setup description file (Python).

    The *experimental setup* file (e.g. ``vipa_2nd.py``) **must** expose at
    least

    ``vars``
        ``dict`` mapping *tunable parameter* names to
        ``[value, min, max]`` triples.  ``min``/``max`` can be ``None`` for a
        Boolean *on/off* switch.

    Optionally it can expose

    ``presets``
        ``dict[str, dict]`` mapping *preset name* → *vars-like dict* that
        overrides *only* the parameters listed in the preset when selected
        via the GUI.  Any parameter omitted from a preset keeps its current
        slider value.  Example::

            sol1 = {
                "V2dX": [0.9381, -2, 2],
                ...
            }
            sol2 = { ... }
            presets = {"Solution 1": sol1, "Solution 2": sol2}

    ``PLOT_TYPE``
        *Optional*, either ``"Z"`` (default) or ``"3D"``.
    """

    # ----------------------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------------------

    def __init__(self, fileName: str, FPS: int = 20):
        self.fileName = fileName
        self.last_update_time = time.time()
        self.FPS = FPS  # target frames per second when moving sliders
        self.render = True  # live‑render flag – can be disabled for speed
        self.changing_preset = False  # flag to suppress slider update events

        # ---------------------------------------------------------------------
        # Load the *experimental setup* file into its own namespace
        # ---------------------------------------------------------------------
        self.namespace: dict[str, object] = {}
        self._create_axes()
        with open(self.fileName, "r", encoding="utf-8") as f:
            self.expsetup_code = f.read()
            exec(self.expsetup_code, self.namespace)

        # Required – tunable parameter specification
        self.tunable_vars_setting: dict[str, list] = self.namespace["vars"]

        # Optional – collection of named presets to offer in the UI
        # Keys become radio‑button labels, values are "vars‑like" dictionaries
        self.presets: dict[str, dict] | None = self.namespace.get("presets")

        # Determine plot type (2‑D Z‑view or 3‑D) requested by the setup file
        self._plot_type = self.namespace.get("PLOT_TYPE", "Z").upper()
        print(f"Plot type requested by setup: {self._plot_type}")

        # For 3‑D mode we need to re‑create *ax0* with projection='3d'
        if self._plot_type == "3D":
            self.ax0.remove()  # discard the 2‑D axes created in _create_axes()
            self.ax0 = plt.subplot(self.gs[0], projection="3d")
            self.namespace["ax0"] = self.ax0

    # ----------------------------------------------------------------------------
    # Figure / axes scaffold
    # ----------------------------------------------------------------------------

    def _create_axes(self):
        """Set up *fig*, *ax0* (main view) and *ax1* side panels."""
        self.fig = plt.figure(figsize=(12, 6))
        # Main optical table panel (ax0) | control / side panels (ax1[0:3])
        self.gs = GridSpec(1, 2, width_ratios=[2.5, 1])
        self.ax0 = plt.subplot(self.gs[0])
        self.gs1 = GridSpecFromSubplotSpec(3, 1, subplot_spec=self.gs[1], hspace=0.3)
        self.ax1 = [plt.subplot(self.gs1[i]) for i in range(3)]
        plt.subplots_adjust(left=0.1, right=0.7)

        # Expose axes to the executed *experimental setup* code
        self.namespace["fig"] = self.fig
        self.namespace["ax0"] = self.ax0
        self.namespace["gs1"] = self.gs1
        self.namespace["ax1"] = self.ax1

    # ----------------------------------------------------------------------------
    # Internal helpers – clearing, slider <→> param conversion
    # ----------------------------------------------------------------------------

    def _clear_axes(self):
        self.ax0.clear()
        for ax in self.ax1:
            ax.clear()

    # --- slider helpers -------------------------------------------------------

    def _get_slider_ax(self, idx: int):
        """Return the *Axes* object that hosts the *idx*‑th slider."""
        # Pack sliders vertically along the right‑hand side of the window.
        return self.fig.add_axes([0.8, 0.95 * (1 - idx / 25), 0.1, 0.03])

    def create_sliders(self):
        """Create (and return) a mapping *name* → Slider/CheckButtons."""
        sliders: dict[str, Slider | CheckButtons] = {}
        for i, (name, param_range) in enumerate(self.tunable_vars_setting.items()):
            slider_ax = self._get_slider_ax(i)
            val, min_val, max_val = param_range
            if min_val is None and max_val is None:
                # Boolean toggle
                sliders[name] = CheckButtons(slider_ax, [name], [bool(val)])
            else:
                sliders[name] = Slider(slider_ax, name, min_val, max_val, valinit=val)
        return sliders

    # Convert current slider values → parameter dict usable by update_table()
    def _sliders_to_params(self, sliders):
        params = {}
        for name, slider in sliders.items():
            if isinstance(slider, Slider):
                params[name] = slider.val
            else:  # CheckButtons
                params[name] = slider.get_status()[0]
        return params

    # Apply a *preset* dict to a *sliders* mapping ---------------------------
    def _apply_preset_to_sliders(self, preset: dict, sliders: dict):
        """Synchronise Slider / CheckButtons widgets with *preset* values."""
        self.changing_preset = True
        for name, values in preset.items():
            if name not in sliders:
                continue  # ignore unknown parameters
            slider = sliders[name]
            target_val, min_val, max_val = values
            if isinstance(slider, Slider):
                # Slider.set_val triggers its on_changed() callback
                slider.set_val(target_val)
            else:  # CheckButtons
                current_val = slider.get_status()[0]
                target_val = bool(target_val)
                if current_val != target_val:
                    # CheckButtons.set_status() does NOT trigger on_clicked()
                    slider.set_active(0, target_val)
        self.changing_preset = False
        # Force a redraw of the figure to reflect the new slider values
        self._clear_axes()
        params = self._sliders_to_params(sliders)
        self.update_table(**params)
        if self.render:
            plt.draw()

    # ----------------------------------------------------------------------------
    # Core rendering / interaction entry‑points
    # ----------------------------------------------------------------------------

    def slider_interactive(self):
        """Launch the GUI with sliders and *optional* preset selector."""
        sliders = self.create_sliders()

        # ------------------------------------------------------------------
        # *update* – called whenever ANY slider value changes
        # ------------------------------------------------------------------
        def _update():
            self._clear_axes()
            params = self._sliders_to_params(sliders)
            self.update_table(**params)
            if self.render:
                plt.draw()

        def update(_):
            # Enforce FPS limit to keep heavy optics drawing responsive
            t_now = time.time()
            if t_now - self.last_update_time < 1 / self.FPS:
                return
            self.last_update_time = t_now

            if not self.changing_preset:
                _update()

        # Connect callbacks ------------------------------------------------
        for slider in sliders.values():
            if isinstance(slider, Slider):
                slider.on_changed(update)
            else:  # CheckButtons
                slider.on_clicked(update)

        # ------------------------------------------------------------------
        # Finetune multiplier (x1 / x10 / x100) – existing functionality
        # ------------------------------------------------------------------
        finetune_multiplier_ax = self.fig.add_axes([0.78, 0.03, 0.15, 0.12])
        finetune_multiplier = RadioButtons(
            finetune_multiplier_ax,
            ["x1", "x10", "x100", "x1000"],
            active=0,
            radio_props={"s": [50, 50, 50, 50]},
        )
        for side in ("top", "right", "left", "bottom"):
            finetune_multiplier_ax.spines[side].set_visible(False)
        self.fig.text(0.78, 0.16, "Finetune Multiplier", fontsize=12, weight="bold")

        def _retune(label):
            # Narrow (or widen) each slider range around its current value
            for i, (name, rng) in enumerate(self.tunable_vars_setting.items()):
                slider = sliders[name]
                if not isinstance(slider, Slider):
                    continue  # Boolean toggle – unchanged
                cur = slider.val
                vmin, vmax = rng[1:]
                span = vmax - vmin
                if label == "x1":
                    new_min, new_max = vmin, vmax
                elif label == "x10":
                    new_min = max(cur - span / 10, vmin)
                    new_max = min(cur + span / 10, vmax)
                elif label == "x100":
                    new_min = max(cur - span / 100, vmin)
                    new_max = min(cur + span / 100, vmax)
                elif label == "x1000":
                    new_min = max(cur - span / 1000, vmin)
                    new_max = min(cur + span / 1000, vmax)

                # Re‑instantiate Slider with the new limits
                slider.disconnect_events()
                slider.ax.remove()
                new_ax = self._get_slider_ax(i)
                sliders[name] = Slider(new_ax, name, new_min, new_max, valinit=cur)
                sliders[name].on_changed(update)
            self.fig.canvas.draw_idle()

        finetune_multiplier.on_clicked(_retune)

        # ------------------------------------------------------------------
        # *NEW* – PRESET SELECTOR -------------------------------------------------
        # ------------------------------------------------------------------
        if self.presets:
            PRESET_X, PRESET_Y = 0.01, 0.8
            preset_ax = self.fig.add_axes([PRESET_X, PRESET_Y, 0.05, 0.1])
            preset_idx = 0
            for i, (name, preset) in enumerate(self.presets.items()):
                if preset == self.tunable_vars_setting:
                    preset_idx = i
                    break
            preset_selector = RadioButtons(
                preset_ax,
                list(self.presets.keys()),
                active=preset_idx,
                radio_props={"s": [50 for _ in self.presets]},
            )
            for side in ("top", "right", "left", "bottom"):
                preset_ax.spines[side].set_visible(False)
            self.fig.text(
                PRESET_X, PRESET_Y + 0.1, "Presets", fontsize=12, weight="bold"
            )

            def _load_preset(label):
                preset = self.presets[label]
                # Update sliders *without* re‑drawing each individual change.
                # Temporarily suppress draw‑events, then call update() once.
                # ----------------------------------------------------------------
                # NOTE: Slider.set_val triggers update() anyway, but we keep the
                # explicit call to ensure the optics model is refreshed even if
                # all preset values match current slider positions.
                self._apply_preset_to_sliders(preset, sliders)
                update(None)

            preset_selector.on_clicked(_load_preset)

        # Initial draw ------------------------------------------------------
        update(None)
        plt.show()

        # On window close, print final parameter dictionary to stdout
        final_params = self._sliders_to_params(sliders)
        print("Final slider values:")
        for k, v in final_params.items():
            print(f"  {k} = {v}")

    # ----------------------------------------------------------------------------
    # Non‑interactive helpers (unchanged from original file)
    # ----------------------------------------------------------------------------

    def update_table(self, **params):
        for key, value in params.items():
            self.namespace[key] = value
        exec(self.expsetup_code, self.namespace)

    def optimize(self, maximize=False):
        initial_values = [v[0] for v in self.tunable_vars_setting.values()]
        bounds = [(v[1], v[2]) for v in self.tunable_vars_setting.values()]
        param_names = list(self.tunable_vars_setting.keys())
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
            cost = float(self.namespace["cost_func"])
            print(f"Iter {n}: cost={cost:.5g}; vals={vals}")
            return -cost if maximize else cost

        result = minimize(
            cost_function, initial_values, bounds=bounds, method="Nelder-Mead"
        )
        print("Optimized parameters:")
        for name, val in zip(param_names, result.x):
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
        for j, yval in enumerate(values_y):
            for i, xval in enumerate(values_x):
                self.update_table(**{param_x: xval, param_y: yval})
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


# -----------------------------------------------------------------------------
# Simple CLI / script entry‑point – mirrors original behaviour
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    FILE_NAME = os.path.join(os.path.dirname(__file__), "../demo", "vipa_2nd.py")
    MODE = "interact"  # 'interact' | 'optimize' | 'scan'

    table = InteractiveOpticalTable(fileName=FILE_NAME)
    table.render = False  # for interactive use we'll enable live‑render ourselves

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
