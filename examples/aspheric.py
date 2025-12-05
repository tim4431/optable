import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

sys.path.append("../")
from src import *

PLOT_TYPE = "Z"  # "Z" or "3D"
# PLOT_TYPE = "3D"  # "Z" or "3D"

ripa_2nd_demo = {
    "x": [0, -5, 5],
}
presets = {
    "default": ripa_2nd_demo,
}

vars = ripa_2nd_demo
for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")


if __name__ == "__main__":
    # plot3d
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0], projection="3d" if PLOT_TYPE == "3D" else None)
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    ax1 = [plt.subplot(gs1[i]) for i in range(3)]

# print(N)
dy = 0
dthetay = 0
N = 3
D = 0.3
R2wl = 780e-7  # in cm
R2w0 = 61e-4  # in cm
CASE = 0
if CASE == 0:
    r0 = [
        Ray([-10, i * D, 0], [1, 0, 0], wavelength=R2wl, w0=R2w0, id=int(i + N))
        .Propagate(-10)
        .TY(dy)
        ._RotAroundLocal([0, 0, 1], [10, 0, 0], dthetay)
        .TX(x)
        for i in np.arange(-N, N + 1)
    ]

F = 20
asp = ASphericExactSphericalLens([F, 0, 0], EFL=F, CT=0.5, diameter=2.54, n=3).RotZ(
    np.pi
)
asp1 = ASphericExactSphericalLens([3 * F, 0, 0], EFL=F, CT=0.5, diameter=2.54, n=3)
# asp = Lens([F, 0, 0], focal_length=F, radius=2.54)
# asp1 = Lens([3 * F, 0, 0], focal_length=F, radius=2.54)

table = OpticalTable()
table.add_components([asp, asp1])
table.ray_tracing(r0)
table.render(ax0, type=PLOT_TYPE, gaussian_beam=True, roi=[-5, 4 * F + 5, -5, 5, -5, 5])

if __name__ == "__main__":
    plt.show()
