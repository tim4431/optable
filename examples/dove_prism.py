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

sol0 = {
    "RLY": [0.0, -np.pi, np.pi],
    "tX": [0.0, -0.1, 0.1],
    "tZ": [0.0, -0.1, 0.1],
    "TX": [0, -1, 1],
    "TZ": [0, -1, 1],
    # "d2F": [1.58, -5, 5],
}
vars = sol0
for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")

L = 6.34
D = 1.515
Ng = 1.515

dp = DovePrism([0, 0, 0], L=L, D=D, Ng=Ng)
z0 = dp.z0

raysx = [
    Ray([3, y, z + 1], [-1, 0, -0.3])
    for y in np.linspace(-1, 1, 7)
    for z in np.linspace(-1, 1, 7)
]
raysz = [
    Ray([x, y, 3], [-0.3, 0, -1])
    for x in np.linspace(-1, 1, 7)
    for y in np.linspace(-1, 1, 7)
]
raysy = [
    Ray([x, 3, z], [0, -1, 0])
    for x in np.linspace(-1, 1, 7)
    for z in np.linspace(-1, 1, 7)
]
rays1d = [Ray([x, -50, z0], [0, 1, 0]) for x in np.linspace(-0.6, 0.6, 7)]
rays0 = [Ray([0, -50, z0], [0, 1, 0])]

# rays = raysx + raysz + raysy
rays = rays0

dp = dp.RotX(tX).RotZ(tZ).TX(TX).TZ(TZ).RotYAroundLocal([0, 0, D / 2], theta=RLY)

DL = -10
FL = 20

mon0 = Monitor([0, 10, z0], width=2 * D, height=2 * D, name="Monitor 0").RotZ(np.pi / 2)

components = [dp]

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
