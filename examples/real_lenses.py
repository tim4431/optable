# vipa_1st.py
# unit in cm

import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

sys.path.append("../")
from src import *

PLOT_TYPE = "Z"  # "Z" or "3D"
# PLOT_TYPE = "3D"  # "Z" or "3D"

if __name__ == "__main__":
    # plot3d
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0], projection="3d" if PLOT_TYPE == "3D" else None)
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    ax1 = [plt.subplot(gs1[i]) for i in range(3)]

demo = {"X1": [30, 0, 60], "X2": [50, 0, 90], "X3": [30, 0, 60], "dX": [-18, -30, 30]}

presets = {
    "default": demo,
}

vars = demo
for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")


N = 3
D = 1.0
R2wl = 780e-7  # in cm
R2w0 = 61e-4  # in cm
r0 = [
    Ray([-10, i * D, 0], [1, 0, 0], wavelength=R2wl, w0=R2w0, id=int(i + N)).Propagate(
        -10
    )
    for i in np.arange(-N, N + 1)
]

#
F = 30
# Meniscus Lens, Thorlabs LE4984
EFLr = 30
R1 = 9.76
R2 = 32.76
CT = 0.54
DL = 3 * 2.54
#
R2l0r = BiConvexLens(
    [X1, 0, 0],
    EFL=EFLr,
    CT=CT,
    R1=R1,
    R2=R2,
    diameter=DL,
    name="L0",
).TX(dX)
R2l2r = BiConvexLens([X1 + X2, 0, 0], CT=1, R1=10, R2=-30, diameter=DL, EFL=30)
R2l1r = (
    BiConvexLens(
        [X1 + 2 * EFLr, 0, 0],
        EFL=EFLr,
        CT=CT,
        R1=R1,
        R2=R2,
        diameter=DL,
        name="L1",
    )
    .RotZ(np.pi)
    .TX(dX)
)

Mon0 = Monitor([2 * F, 0, 0], width=2, height=2, name="Monitor 0")
Mon1 = Monitor([4 * F, 0, 0], width=2, height=2, name="Monitor 1")

table = OpticalTable()
components = [R2l0r, R2l1r]
# components = [R2l0r, R2l2r]
table.add_components(components)
monitors = [Mon0, Mon1]
table.add_monitors(monitors)
table.ray_tracing(r0)
table.render(
    ax0, type=PLOT_TYPE, roi=(-3, 4 * F + 10, -4, 4, -1, 1), gaussian_beam=True
)
Mon0.render_scatter(ax1[0], annote_delta_pos=True)
Mon1.render_scatter(ax1[1], annote_delta_pos=True)
table.render(
    ax1[2],
    type=PLOT_TYPE,
    roi=(-2, 2, -1, 1, -1, 1),
    gaussian_beam=True,
    detailed_render=True,
)


if __name__ == "__main__":
    # calculate_abcd()
    plt.show()
