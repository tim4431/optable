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


sol0 = {"y": [0.0, -1, 1], "z": [0.0, -1, 1]}

vars = sol0
for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")

theta = 0.012
L = 4
wl = 780e-7
w0 = 61e-4
nmat = Glass_NBK7()
n = nmat.n(780e-9)
alpha = n * theta / (n**2 - 1)
print(f"alpha: {alpha*180/np.pi} deg")

rays = [
    Ray([3, y + L / 2, 0], [-1, 0, 0], wavelength=wl, w0=w0, id=i)
    .Propagate(-10)
    .RotZ(theta)
    for i, y in enumerate(np.linspace(-1, 1, 5))
]
print([ray._id for ray in rays])

dth = 0.05
ps = TriangularPrism(
    origin=[0, 0, 0],
    width=L,
    height=L,
    n1=1,
    n2=n,
    alpha=np.pi / 4 + dth,
    beta=np.pi / 2 - alpha - 2 * dth,
    reflectivity_1=1,
    reflectivity_3=0.01,
    max_interact_count_2=10,
    max_interact_count_3=10,
)

mon0 = Monitor([-3, 0, 0], width=L, height=L, name="Monitor 0")

components = [ps]

table = OpticalTable()
table.add_components(components)
table.add_monitors([mon0])
table.ray_tracing(rays)

table.render(ax0, type=PLOT_TYPE, roi=[-5, 5, -10, 5, -5, 5], gaussian_beam=True)
mon0.render_scatter(ax1[0])

if __name__ == "__main__":
    # plt.axis("off")
    # plt.savefig("gaussian_beam.png", dpi=300, bbox_inches="tight")
    plt.show()
