# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package identity

- PyPI distribution name: `optical-table`. Import name: `optable`. The repo root and the package directory share the name `optable/` — be careful with paths.
- `from optable import *` re-exports everything from `base`, `component_group`, `material`, `monitor`, `optical_component`, `optical_table`, `ray`, `solver`, `surfaces`. Example scripts rely on this star-import.

## Common commands

- Editable install: `pip install -e .` (the published package is built via `python -m build`).
- Run an example: `python examples/gaussian_beam.py` (many example scripts `savefig` to `../docs/` with relative paths, so run from [examples/](examples/) or [demo/](demo/) — `.vscode/launch.json` sets `cwd=${fileDirname}` to match).
- Build docs locally: `sphinx-build docs docs/_build`. Setting `OPTABLE_BUILD_EXAMPLES=1` forces example scripts to execute during the build (default is "only on Read the Docs"); [docs/conf.py](docs/conf.py) wires this into the `builder-inited` hook via [docs/build_examples_outputs.py](docs/build_examples_outputs.py) and [docs/generate_examples_docs.py](docs/generate_examples_docs.py).
- There is no test suite and no lint/format configuration — don't fabricate `pytest`/`ruff` commands.
- Release: [.github/workflows/python-publish.yml](.github/workflows/python-publish.yml) publishes to PyPI when a commit on `main` has a message starting with `v` (e.g. `v1.0.3 ...`). Version strings live in [pyproject.toml](pyproject.toml); [optable/__init__.py](optable/__init__.py) reads the installed distribution's version via `importlib.metadata`.

## Architecture

The code is a small ray-tracing engine with a matplotlib-driven GUI layered on top. Layers, bottom up:

1. **Geometry primitives** — [optable/base.py](optable/base.py) defines `Base` (id-preserving deep-copy) and `Vector` (3D origin + `transform_matrix`, with `R`, `RotX/Y/Z`, `Translate`). Every scene object inherits `_id` so that ray copies share identity across interactions; `norender_set` on `OpticalTable` uses these ids.
2. **Surfaces** — [optable/surfaces.py](optable/surfaces.py) exposes `Surface` subclasses (`Plane`, `Rectangle`, `Spherical`, etc.) with `f(P)=0`, `normal(P)`, `within_boundary(P)`, `parametric_boundary_3d(t)`, and bbox merging. Flag `planar` short-circuits the intersection solver.
3. **Numerics** — [optable/solver.py](optable/solver.py) has `solve_ray_bboxes_intersections` (vectorised slab test) used for early rejection. Non-planar surface intersections use `scipy.optimize.brentq` bracketing inside `OpticalComponent._find_zero`.
4. **Materials & refractive index** — [optable/material.py](optable/material.py) provides `RefractiveIndex` (a descriptor) and a `Glass` catalog. Ray `_n` is attached via this descriptor, so wavelength-dependent dispersion happens per-ray.
5. **Ray + Gaussian beam** — [optable/ray.py](optable/ray.py): `Ray` carries origin/direction/intensity/wavelength, plus an optional complex Gaussian `q` (`qo`, derived from `w0`). `GaussianBeam` is a namespace of static helpers (`q_at_waist`, `spot_size`, `rayleigh_range`, `radius_of_curvature`). `_pathlength` accumulates optical path length during tracing.
6. **Components** — [optable/optical_component.py](optable/optical_component.py) defines `OpticalComponent` (a `Vector` with a `surface`, `transform_matrix`, local-frame interaction, bbox, interact-count cap). Concrete classes: `Mirror`, `Lens`, `Refractive`, `Misc`, and the beam-splitter / grating variants. Convention: the component's **normal is local +X** (`tangent_Y = +Y`, `tangent_Z = +Z`), and lab-frame vectors are obtained by `transform_matrix @ local`.
7. **Component groups** — [optable/component_group.py](optable/component_group.py) bundles multiple components, monitors, and named "refpoints" into one object that transforms as a rigid body. Concrete groups: `Asphere`, `GlassPlate`, `Lens` (thick), `MMA`, `Prism`. Rotations use `_RotAroundLocal` so child rays, components, and refpoints stay consistent.
8. **Monitors** — [optable/monitor.py](optable/monitor.py) `Monitor` is a rectangular component that records `(P, intensity, ray)` on hit, with sort-by-YZ or sort-by-ID accessors. Used for post-processing and ABCD extraction.
9. **Scene** — [optable/optical_table.py](optable/optical_table.py) `OpticalTable` owns components/monitors/rays. `ray_tracing` dispatches to `_single_ray_tracing`: a BFS where each component's `interact(ray)` returns `(t, new_rays)`; the smallest positive `t` wins, and `new_rays` are re-queued if `alive`, archived otherwise. Safety limits are `MAX_TRACE_NUM=2000` and `MAX_TRACEING_TIME=600s`, overridable via `perfomance_limit`. `render(ax, type=...)` supports 2D projections `"X"|"Y"|"Z"` and `"3D"`, with `roi=[xmin,xmax,ymin,ymax,(zmin,zmax)]`, `gaussian_beam=True`, `spot_size_scale`, etc. `calculate_abcd_matrix(mon0, mon1, rays)` extracts a per-ray 2×2 matrix via finite-difference perturbation of position and angle at `mon0`.
10. **Interactive GUI** — [optable/interact.py](optable/interact.py) (and the newer [optable/interact_new.py](optable/interact_new.py)) implement `InteractiveOpticalTable`, a matplotlib Slider/CheckButtons/Radio panel that `exec`s a user setup script in its own namespace. [optable/setup_runtime.py](optable/setup_runtime.py) wraps the exec with `os.chdir(setup_dir)` and a patched `np.load` so the script resolves relative paths against its own location.
11. **Zemax I/O** — [optable/zmxread.py](optable/zmxread.py) and [optable/zmxro.py](optable/zmxro.py) parse Zemax `.zmx` lens prescriptions into component groups.

## Experiment-script contract (for the interactive GUI)

When writing or editing files driven by `InteractiveOpticalTable`, the script must define these at module scope:

- `vars: dict[str, [value, vmin, vmax]]` — each entry becomes a slider. If both `vmin` and `vmax` are `None`, it's rendered as a `CheckButtons` toggle (boolean) and excluded from optimization.
- `presets: dict[str, dict]` *(optional)* — named parameter-override sets; unspecified keys inherit current slider state.
- `PLOT_TYPE = "Z" | "3D"` *(optional, default `"Z"`)* — chooses 2D projection vs 3D axes.
- Any callable named `cost_<name>` (or numeric variable with that prefix) is auto-discovered as an optimization target; it is called with no arguments and must return a float. The first one discovered becomes the default.
- `fig`, `ax0`, `ax1`, `gs1` are injected into the script's namespace before execution — don't create your own figure.

## Coordinate & unit conventions

- Default unit is `1e-2` (cm); it flows through `Vector.__init__` and `OpticalTable.__init__`. Wavelengths in example scripts are in SI meters (e.g. `780e-9`), so mixing scales is intentional — Gaussian beam math uses raw `wavelength` and distances in the same unit the scene uses.
- Local frame: normal is +X, so `Mirror([0,0,0]).RotZ(π/6)` rotates the surface normal in the XY plane. The default "Z" render projects onto the XY plane; "X"/"Y" projections swap the visible axes.
- `_id` is preserved across copies of a ray/component — do not generate new ids when cloning; rely on `Base.copy()`.

## Gotchas

- Example scripts call `plt.savefig("../docs/...")` with paths relative to the script's own directory. Run from the script's directory (VS Code's `cwd=${fileDirname}` already does this) or the save will land somewhere unexpected.
- `OpticalTable.ray_tracing` **appends** to `self.rays`; call it on a fresh table or clear `self.rays` / monitors between runs (see `calculate_abcd_matrix`'s internal `_simulate` for the pattern).
- `component.should_interact(ray_id)` + `max_interact_count` is used to break infinite cavity bounces — respect it when writing new component classes.
