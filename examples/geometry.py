import numpy as np, sys, matplotlib.pyplot as plt

sys.path.append("../")
from src import *

if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(111, projection="3d")


m0 = Mirror([0, 0, 0])
block = Block([0, 0, 0], 1, 1, 0.5, 0.5)

# Summary, rays, components, monitors
rays = []
components = [m0, block]
monitors = []

table = OpticalTable()
table.add_components(components)
table.add_monitors(monitors)
table.ray_tracing(rays)

# ax0.annotate(mon0.ndata, (mon0.origin[0], mon0.origin[1]), fontsize=15, color="black")
table.render(
    ax0,
    type="3D",
)

if __name__ == "__main__":
    # plt.axis("off")
    # plt.savefig("gaussian_beam.png", dpi=300, bbox_inches="tight")
    plt.show()
