# Optable

A simple ray tracing and visualization tool for freespace optics.

dedicated to phys students working with freespace optics.

[![GitHub stars](https://img.shields.io/github/stars/tim4431/optable?style=for-the-badge)]()

## Example
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