import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from optable import *

# PLOT_TYPE = "3D"  # "Z" or "3D"
PLOT_TYPE = "Z"  # "Z" or "3D"

if __name__ == "__main__":
    # plot3d
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0], projection="3d" if PLOT_TYPE == "3D" else None)
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    ax1 = [plt.subplot(gs1[i]) for i in range(3)]


# Tunable parameters = [value, min, max]
sol1 = {"Z": [0, -10, 10]}

vars = sol1

for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")

# 0
rays = [
    Ray(
        [-3, 2, 0],
        [np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
        wavelength=780e-7,
        w0=2e-4,
    ).Propagate(-2),
]
L = 2
D = 0.5

gs = GlassSlab(
    [0, 0, 0], width=L, height=L, thickness=D, n1=1, n2=1.5, reflectivity=0.2
)
components = [gs]
table = OpticalTable()
table.add_components(components)
table.ray_tracing(rays)

table.render(
    ax0,
    type=PLOT_TYPE,
    gaussian_beam=True,
    roi=(-5, 15, -5, 5, -5, 5),
    detailed_render=True,
)

if __name__ == "__main__":
    if PLOT_TYPE == "3D":
        ax0.view_init(elev=30, azim=-150)
        ax0.grid(False)
        ax0.set_axis_off()
    #
    # table.export_rays_csv("1dbeams_rays_traced.csv")
    # table.export_components_csv(
    #     "1dbeams_components.csv",
    #     avoid_flatten_classname=[],
    #     ignore_classname=["Block"],
    # )
    plt.show()
