import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

sys.path.append("../")
from src import *

PLOT_TYPE = "3D"
if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0], projection="3d")
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    ax1 = [plt.subplot(gs1[i]) for i in range(3)]

sol0 = {"y": [0.0, -1, 1], "z": [0.0, -1, 1]}

vars = sol0
for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")

rays = [Ray([-10, y, z], [1, 0, 0])]

L = 2

cb = MirrorCube([0, 0, 0], D=L).RotY(np.pi)

mon0 = Monitor([-3, 0, 0], width=L, height=L, name="Monitor 0")

components = [cb]

table = OpticalTable()
table.add_components(components)
table.add_monitors([mon0])
table.ray_tracing(rays)

table.render(
    ax0,
    type="3D",
    roi=[-5, 5, -5, 5, -5, 5],
)
mon0.render_scatter(ax1[0])

if __name__ == "__main__":
    # plt.axis("off")
    # plt.savefig("gaussian_beam.png", dpi=300, bbox_inches="tight")
    plt.show()
