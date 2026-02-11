import numpy as np, sys, matplotlib.pyplot as plt
from optable import *

if __name__ == "__main__":
    # fig, ax0 = plt.subplots(1, 1, figsize=(12, 6))
    # 3d
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(111, projection="3d")


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

# ax0.annotate(mon0.ndata, (mon0.origin[0], mon0.origin[1]), fontsize=15, color="black")
table.render(
    ax0,
    # type="Z",
    type="3D",
    gaussian_beam=True,
    spot_size_scale=1,
    # roi=[-15, 30, -10, 20],
    roi=[-15, 30, -10, 20, -10, 10],
)

if __name__ == "__main__":
    # plt.axis("off")
    plt.savefig("../docs/gaussian_beam_3d.png", dpi=300, bbox_inches="tight")
    plt.show()
