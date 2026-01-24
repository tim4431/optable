import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from scipy.optimize import minimize

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


# Thorlabs LA1353 - 75mm
EFLr = 20
CT = 1.01
Rr = 10.3
DL = 7.5
ROTL0 = False
R2l0r = PlanoConvexLens(
    [EFLr, 0, 0],
    EFL=EFLr,
    CT=CT,
    diameter=DL,
    R=Rr,
    name="L0",
)
N = 3
D = 0.5
R2wl = 780e-7  # in cm
R2w0 = 61e-4  # in cm
r0 = [
    Ray([-10, i * D, 0], [1, 0, 0], wavelength=R2wl, w0=R2w0, id=int(i + N)).Propagate(
        -10
    )
    for i in np.arange(-N, N + 1)
]
rays = multiplex_rays_in_wavelength(r0, [780e-7, 560e-7, 1230e-7])

table = OpticalTable()
table.add_components([R2l0r])
table.ray_tracing(rays)
table.render(ax0, type="Z", roi=[-5, 2 * EFLr + 5, -5, 5], gaussian_beam=True)

if __name__ == "__main__":
    plt.show()
