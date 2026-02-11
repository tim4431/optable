import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from optable import *

PLOT_TYPE = "Z"
# PLOT_TYPE = "3D"

if __name__ == "__main__":
    # plot3d
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0], projection="3d" if PLOT_TYPE == "3D" else None)
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    ax1 = [plt.subplot(gs1[i]) for i in range(3)]


wl = 780e-9
w0 = 10e-6


r0 = [
    Ray([-10, 0, 0], [1, 0, 0], wavelength=wl, w0=w0),
    Ray([-10, 2, 0], [1, 0, 0], wavelength=wl, w0=w0),
    Ray([-10, 4, 0], [1, 0, 0], wavelength=wl, w0=w0),
    Ray([-10, 6, 0], [1, 0, 0], wavelength=wl, w0=w0),
    Ray([-10, 9, 0], [1, 0, 0], wavelength=wl, w0=w0),
    Ray([-10, 21, 0], [1, 0, 0], wavelength=wl, w0=w0).RotZ(-np.pi / 4),
]

m0 = Mirror([0, 0, 0]).RotZ(np.pi / 6)
l0 = Lens([0, 2, 0], radius=0.8, focal_length=5)
l1 = Lens([0, 4, 0], radius=0.8, focal_length=10)
l2 = Lens([0, 6.5, 0], radius=0.8, focal_length=10)
slab0 = GlassSlab([0, 9, 0], n1=1, n2=2, thickness=5)
m1 = Mirror([0, 11, 0]).RotZ(-np.pi / 2)

# Summary, rays, components, monitors
rays = r0
components = [m0, l0, l1, l2, slab0, m1]
monitors = []

table = OpticalTable()
table.add_components(components)
table.add_monitors(monitors)
table.ray_tracing(rays)

table.render(
    ax0,
    type=PLOT_TYPE,
    gaussian_beam=True,
    spot_size_scale=1,
    roi=[-15, 30, -10, 20, -10, 10],
)

if __name__ == "__main__":
    # plt.axis("off")
    plt.savefig("../docs/gaussian_beam.png", dpi=300, bbox_inches="tight")
    plt.show()
