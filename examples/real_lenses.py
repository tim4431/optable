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

demo = {
    "dy": [0.0, -0.1, 0.1],
    "dthetay": [0.0, -0.1, 0.1],
}

presets = {
    "default": demo,
}

vars = demo
for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")


N = 2
D = 0.1
R2wl = 780e-7  # in cm
R2w0 = 61e-4  # in cm
# r0 = [
#     Ray([-10, i * D, 0], [1, 0, 0], wavelength=R2wl, w0=R2w0, id=int(i + N))
#     .Propagate(-10)
#     .TY(dy)
#     ._RotAroundLocal([0, 0, 1], [10, 0, 0], dthetay)
#     for i in np.arange(-N, N + 1)
# ]
R2DMLA = 360e-4
R2NMLA = 20
R2L = R2DMLA * R2NMLA / 2
R2MLAroc = 3.0
R2d = R2MLAroc / 2
R2theta0 = np.arctan(R2DMLA / R2d / 2)
r0 = [
    Ray([-10, R2L, 0], [1, 0, 0], wavelength=R2wl, w0=R2w0, id=0)
    .Propagate(-10)
    ._RotAround([0, 0, 1], [0, R2L, 0], -R2theta0),
]

# Thorlabs LA1634-B
EFLr = 35
CT = 0.48
Rr = 18.09
DL = 2 * 2.54
F = EFLr
#
R2l0r = PlanoConvexLens(
    [F, 0, 0],
    EFL=F,
    CT=CT,
    diameter=DL,
    R=Rr,
    name="L0",
).RotZ(np.pi)

R2l1r = PlanoConvexLens(
    [3 * F, 0, 0],
    EFL=F,
    CT=CT,
    diameter=DL,
    R=Rr,
    name="L1",
)

Mon0 = Monitor([2 * F, 0, 0], width=2, height=2, name="Monitor 0")
Mon1 = Monitor([4 * F, 0, 0], width=2, height=2, name="Monitor 1")

table = OpticalTable()
components = [R2l0r, R2l1r]
table.add_components(components)
monitors = [Mon0, Mon1]
table.add_monitors(monitors)
table.ray_tracing(r0)
table.render(ax0, type=PLOT_TYPE, roi=(-3, 4 * F + 3, -3, 3, -1, 1), gaussian_beam=True)
Mon0.render_scatter(ax1[0], annote_delta_pos=True)
Mon1.render_scatter(ax1[1], annote_delta_pos=True)
table.render(
    ax1[2],
    type=PLOT_TYPE,
    roi=(-2, 2, -1, 1, -1, 1),
    gaussian_beam=True,
    detailed_render=True,
)
yList = []
tyList = []
for ray in r0:
    hit_point = Mon1.get_ray_id(ray._id)[1]
    yList.append(hit_point[0])
    tyList.append(hit_point[2])


def calculate_abcd():
    # Calculate ABCD matrix numerically for each ray in r0
    # return a list of ABCD matrix elements
    dy = 1e-5
    dthetay = 1e-5
    y0List, ty0List = simulate(0, 0)
    y1List, ty1List = simulate(dy, 0)
    y2List, ty2List = simulate(0, dthetay)
    ABCDList = []
    for i in range(len(y0List)):
        y0 = y0List[i]
        ty0 = ty0List[i]
        y1 = y1List[i]
        ty1 = ty1List[i]
        y2 = y2List[i]
        ty2 = ty2List[i]
        A = (y1 - y0) / dy
        B = (y2 - y0) / dthetay
        C = (ty1 - ty0) / dy
        D = (ty2 - ty0) / dthetay
        ABCDList.append((A, B, C, D))
    for i, (A, B, C, D) in enumerate(ABCDList):
        print(f"Ray {i}: ABCD = [{A:.6f}, {B:.6f}; {C:.6f}, {D:.6f}]")


if __name__ == "__main__":
    # calculate_abcd()
    plt.show()
