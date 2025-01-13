# Optable

A simple ray tracing and visualization tool for freespace optics.

dedicated to phys students working with freespace optics.

[![GitHub stars](https://img.shields.io/github/stars/tim4431/optable?style=for-the-badge)](https://github.com/tim4431/optable/stargazers)
[![Wiki](https://img.shields.io/github/stars/tim4431/optable?style=for-the-badge)](https://github.com/tim4431/optable/wiki)

## Example
**1. Micro Lens Array**
```python
# expsetup.py

import numpy as np, sys

sys.path.append("../src/")
from optical_component import *
from component_group import *
from optical_table import OpticalTable
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig, ax0 = plt.subplots(1, 1, figsize=(12, 6))


# Tunable parameters = [value, min, max]
# can be used in src/interact.py
vars = {
    "dt": [0.2, -0.6, 0.6],
}

for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")


# Constants

# Rays
# r0 = Ray([-4, 0, 0], [1, -0.2, 0])
r2d = [Ray([-10, 0, 0], [1, 0, 0]).TY(i) for i in np.linspace(-0.5, 3.5, 30)]

# # Components
mla = MLA([0, 0, 0], N=4, pitch=1, focal_length=10, radius=0.5).RotZ(dt)

# Summary, rays, components, monitors
# rays = [r0]
rays = r2d
components = [mla]
monitors = []

table = OpticalTable()
table.add_components(components)
table.add_monitors(monitors)
table.ray_tracing(rays)


table.render(
    ax0,
    type="Z",
    roi=(-5, 15, -2, 4),
)

if __name__ == "__main__":
    plt.savefig("tilt_MLA.png", dpi=300, bbox_inches="tight")
    plt.show()
```

**Result**
![docs/tilt_MLA.png](docs/tilt_MLA.png)


**2. Gaussian beam tracing**

```python
import numpy as np, sys, matplotlib.pyplot as plt

sys.path.append("../src/")
from optical_component import *
from optical_table import OpticalTable
from component_group import GlassSlab

if __name__ == "__main__":
    fig, ax0 = plt.subplots(1, 1, figsize=(12, 6))


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
    type="Z",
    gaussian_beam=True,
    spot_size_scale=1,
    roi=[-15, 30, -10, 20],
)

if __name__ == "__main__":
    # plt.axis("off")
    plt.savefig("../docs/gaussian_beam.png", dpi=300, bbox_inches="tight")
    plt.show()
```

**Result**
![docs/gaussian_beam.png](docs/gaussian_beam.png)
