import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

sys.path.append("../")
from src import *

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
theta = 0.1
rays = [
    Ray(
        [-2, y, z],
        [np.cos(theta), -np.sin(theta), 0],
        wavelength=780e-7,
        w0=10e-4,
    ).Propagate(-5)
    for y in np.linspace(-0.2, 0.2, 5)
    for z in [0]
]
L = 2
D = 0.5

# nbk7 = Glass_NBK7()
nbk7 = Glass_UVFS()
n0 = nbk7.n(1064e-9)
theta0 = (360e-6) / (2 * 1.5e-2)
alpha = theta0 * (n0 / (n0**2 - 1))
print(alpha * 180 / np.pi)
# gs = GlassSlab(
#     [0, 0, 0], width=L, height=L, thickness=D, n1=1, n2=1.5, reflectivity=0.0
# )
n = 1.5
wp = WedgePlate(
    [0, 0, 0],
    width=L,
    height=L,
    thickness=D,
    wedge_angle=-theta * n / (n**2 - 1),
    n1=Vacuum(),
    n2=ConstMaterial(n=n),
)
mon0 = Monitor([3.17, -0.33, 0], width=2, height=2).RotZ(-np.arcsin(0.03960814))
components = [wp]
table = OpticalTable()
table.add_components(components)
table.add_monitors([mon0])
table.ray_tracing(rays)

print(mon0.directionList)
s = mon0.get_waist_distance()
plt.figure()
plt.plot(s * 1e2)
plt.xlabel("Propagation Distance (cm)")
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
    table.export_rays_csv("1dbeams_rays_traced.csv")
    table.export_components_csv(
        "1dbeams_components.csv",
        avoid_flatten_classname=[],
        ignore_classname=["Block"],
    )
    plt.show()
