# expsetup.py

import numpy as np, sys

sys.path.append("../src/")
from optical_components import *
from optical_table import OpticalTable
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig, ax0 = plt.subplots(1, 1, figsize=(12, 6))

# Tunable parameters = [value, min, max]
vars = {"dY": [0.5, -1, 1], "dt1": [0.02, -0.2, 0.2], "dt2": [0.02, -0.2, 0.2]}

for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")


# Constants
L = 10 * 4 / 3
D = 4
R = 0.9
# F = 50

# Rays
r0 = Ray([0, 0, 0], [1, 0, 0]).TY(dY)

# Optical components
m0 = Mirror([0, 0, 0], radius=D, reflectivity=R).RotZ(-np.pi / 4)
m1 = Mirror([L, 0, 0], radius=D, reflectivity=R).RotZ(+np.pi / 4 + dt1)
m2 = Mirror([L, -L, 0], radius=D, reflectivity=R).RotZ(-np.pi / 4 + dt2)
m3 = Mirror([0, -L, 0], radius=D, reflectivity=R).RotZ(+np.pi / 4)

# Summary, rays, components, monitors
rays = [r0]
components = [m0, m1, m2, m3]

table = OpticalTable()
table.add_components(components)
r0 = Ray([2, 0, 0], [1, 0, 0])
rays = [r0]
table.ray_tracing(rays)
table.render(ax0, type="Z", roi=(-5, 20, -20, 5))

if __name__ == "__main__":
    # plt.axis("off")
    # plt.savefig("cavity_4mir.png", dpi=300, bbox_inches="tight")

    plt.show()
