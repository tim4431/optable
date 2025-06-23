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

sol0 = {
    "w": [0.5, 0, 3],
    "dY0": [-0.5, -2, 2],
    "dY1": [-0.5, -2, 2],
    "dXMon": [0, -5, 5],
}
sol1 = {
    "w": [0.5, 0, 3],
    "dY0": [0, -2, 2],
    "dY1": [0, -2, 2],
    "dXMon": [0, -5, 5],
}

presets = {
    "biased": sol0,
    "centered": sol1,
}

vars = sol1
for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")

# EDMUND 26-158
# 40.0mm Diameter x 400.0mm FL, 785nm V-Coat, PCX Lens
EFL1 = 40
CT1 = 0.5
DIA1 = 4
R1 = 20.672
DF1 = 0.482
# Thorlab LA1708-780
EFL2 = 19.932
CT2 = 0.28
DIA2 = 2.54
R2 = 10.3
DF2 = 0.2324
#
DRay = 500e-4
N = np.ceil(w / DRay)

r0 = [Ray([-10, y, 0], [1, 0, 0]) for y in np.arange(0, N + 1) * DRay]

l0 = PlanoConvexLens(
    origin=np.array([0, 0, 0]), EFL=EFL1, CT=CT1, diameter=DIA1, R=R1
).TY(dY0)
l1 = (
    PlanoConvexLens(
        origin=np.array([EFL1 + EFL2 + DF1 + DF2, 0, 0]),
        EFL=EFL2,
        CT=CT2,
        diameter=DIA2,
        R=R2,
    )
    .RotZ(np.pi)
    .TY(dY1)
)
Mon0 = Monitor(
    [-5, 0, 0],
    width=2,
    height=2,
)
Mon1 = Monitor(
    [EFL1 + EFL2 + 5, 0, 0.2],
    width=2,
    height=2,
).TY(dY1)
Monf = Monitor(
    [EFL1 + DF1, 0, 0],
    width=0.1,
    height=0.1,
)
components = [l0, l1]
monitors = [Mon0, Mon1, Monf]

table = OpticalTable()
table.add_components(components)
table.add_monitors(monitors)
table.ray_tracing(r0)
table.render(ax0, type=PLOT_TYPE, roi=(-3, 70, -3, 3, -2, 2))
Mon0.render_scatter(ax1[0], annote_delta_pos=True)
Mon1.render_scatter(ax1[1], annote_delta_pos=True)

print(Mon1.tYList.tolist())

if __name__ == "__main__":
    plt.show()
