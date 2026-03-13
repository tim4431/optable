import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from optable import *


PLOT_TYPE = "Z"  # "Z" or "3D"
# PLOT_TYPE = "3D"  # "Z" or "3D"

if __name__ == "__main__":
    # plot3d
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0], projection="3d" if PLOT_TYPE == "3D" else None)
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    ax1 = [plt.subplot(gs1[i]) for i in range(3)]


# Tunable parameter: rotation angle of the mirror pair around the x-axis
vars = {"phi": [0.0, -np.pi, np.pi]}

for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")

L = 4  # distance between the two mirrors along the deflected path
D = 1.5  # mirror radius

# Incoming ray along the x-axis
r0 = Ray([-10, 1, 0], [1, 0, 0])._RotAround([0, 1, 0], [0, 0, 0], 0.1)

mp = MirrorPair([3, 0, 0], 4, 4).RotX(phi)
Mon0 = Monitor([1, 0, 0], 5, 5)
Mon1 = Monitor([-3, 0, 0], 5, 5)
components = [mp]
rays = [r0]

table = OpticalTable()
table.add_components(components)
table.add_monitors([Mon0, Mon1])
table.ray_tracing(rays)
table.render(ax0, type=PLOT_TYPE, roi=[-5, 5, -6, 6, -6, 6])
Mon0.render_scatter(ax1[0])
Mon1.render_scatter(ax1[0])

if __name__ == "__main__":
    plt.show()
