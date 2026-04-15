import numpy as np, sys, os, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec


sys.path.append("../")
from optable import *

# PLOT_TYPE = "X"  # "Z" or "3D"
PLOT_TYPE = "Z"  # "Z" or "3D"
# PLOT_TYPE = "3D"  # "Z" or "3D"

if __name__ == "__main__":
    # plot3d
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0])
    ax0 = [
        plt.subplot(gs0[i], projection="3d" if PLOT_TYPE == "3D" else None)
        for i in range(2)
    ]
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    ax1 = [plt.subplot(gs1[i]) for i in range(3)]

# Tunable parameters = [value, min, max]
# >>> ripa_2nd_demo
ripa_2nd_demo = {
    "R2dXmp": [0, -1, 1],
    "R2dYmp": [0.0, -0.1, 0.1],
}
# <<< ripa_2nd_demo


presets = {
    "default": ripa_2nd_demo,
}

vars = ripa_2nd_demo
for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")


# >>> ripa_2nd
# MMAShift = False
MMAShift = True
MMAReplaceWithMir = False
# MMAReplaceWithMir = True
MMABlock = False
# MMABlock = True
BMMABlock = False
# BMMABlock = True
# EnableMP = False
EnableMP = True
# Constants
BW = 6.8  # in GHZ
R2ultraR = 0.9995
R2R = 0.98
R2wl = 780e-7  # in cm
# R2DMLA = 500e-4
R2DMLA = 260e-4
R2NMLA = 20
R2MMLA = 5
R2W = R2DMLA * R2NMLA / 2
R2H = R2DMLA * R2MMLA / 2

# Rays
R2d = 3e8 / (2 * BW * 1e9) / 0.01
print("R2d=", R2d, "cm")
Lrt = 2 * np.sqrt(R2d**2 + R2DMLA**2 / 4)
# R2MLAROCCOEF = 1.0
R2MLAROCCOEF = 1.2
R2MLAroc = (Lrt) * R2MLAROCCOEF
print("R2MLAroc=", R2MLAroc, "cm")
# R2w0 = np.sqrt(R2wl * R2MLAroc / (2 * np.pi))
R2w0 = np.sqrt(R2wl * (np.sqrt((Lrt) * (R2MLAroc / 2 - Lrt / 4))) / (np.pi))
print("R2w0=", R2w0 * 1e4, "um")
print("R2DMLA/R2w0=", R2DMLA / R2w0)
clipping_loss = np.exp(-2 * ((R2DMLA / 2) ** 2) / ((R2w0) ** 2))
print("clipping_loss=", clipping_loss)
print("MMLA*NMLA*clipping_loss=", R2MMLA * R2NMLA * clipping_loss)
R2theta0 = np.arctan(R2DMLA / R2d / 2)
print("R2theta0=", R2theta0)
R2ZRay = (R2MMLA / 2) * R2DMLA
# print(R2ZRays)
# Input physical rays
R2X_waist = 0
D = 0.1
MMAshify = R2DMLA / (R2MMLA + 1)
vec_input = np.array([R2d, -MMAshify / 2, R2DMLA / 2])
vec_input = vec_input / np.linalg.norm(vec_input)
vec_output = np.array([-R2d, -MMAshify / 2, R2DMLA / 2])
vec_output = vec_output / np.linalg.norm(vec_output)
LMMAconnect = (Lrt / 2) * np.linalg.norm([vec_output[0], vec_output[2]])
vec_output_projxz = [vec_output[0], vec_output[2]]
vec_output_projxz /= np.linalg.norm(vec_output_projxz)

R2rays0 = [
    Ray(
        [0, R2W, -R2ZRay],
        vec_input,
        wavelength=R2wl,
        w0=R2w0,
    ).Propagate(-(R2X_waist - (0)))
]
origin_output = np.array([0, R2ZRay])
DMMA = 0.1


def hit_point(t):
    p = origin_output + vec_output_projxz * t
    o_BMMA = np.array([-DMMA, 0])
    vec2 = p - o_BMMA
    t2 = np.linalg.norm(vec2)
    vec_mid = (vec2 / t2 + vec_output_projxz) / 2
    angle = np.arcsin(vec_mid[1])
    return t + t2, p, angle


# solve t within 0-Lrt st hit_point(t)=Lrt/2
from scipy.optimize import brentq

t_solution = brentq(lambda t: hit_point(t)[0] - LMMAconnect, 0, Lrt)
_, p, angle_solution = hit_point(t_solution)

R2delta = 2 * R2w0  # cut on both sides of mirror
R2Wm0 = (R2NMLA) * R2DMLA + 2 * R2delta
R2Hm0 = (R2MMLA) * R2DMLA - 2 * R2delta

R2mp1 = SquareMirror(
    [p[0], 0, p[1]], width=R2Wm0, height=0.1, reflectivity=R2ultraR
).RotY(angle_solution)
R2mp2 = SquareMirror(
    [p[0], 0, -p[1]], width=R2Wm0, height=0.1, reflectivity=R2R, transmission=0.1
).RotY(-angle_solution)


BMMAshift = -(R2DMLA / 2) * (R2MMLA) / (R2MMLA + 1)
R2m0 = MMA(
    origin=[-DMMA, 0, 0],
    N=(R2NMLA, 1),
    pitch=R2DMLA,
    roc=R2MLAroc,
    n=1.5,
    thickness=DMMA,
    reflectivity=R2ultraR,
    transmission=0,
    mma_width=R2Wm0,
    mma_height=R2Hm0,
    mma_shifty=BMMAshift,
    back_transmission=0,
    back_reflectivity=1,
)

R2XMLA = R2d
R2mma = MMA(
    origin=[R2XMLA, 0, 0],
    N=(R2NMLA, R2MMLA),
    pitch=R2DMLA,
    roc=R2MLAroc,
    n=1.5,
    thickness=0.1,
    reflectivity=R2ultraR,
    transmission=0.0,
    render_comp_vec=False,
    name="R2MMA",
    shifty_z=0 if not MMAShift else -MMAshify,
).TY(R2DMLA / 2 - MMAshify / 2)

R2m1 = SquareMirror(origin=[R2XMLA, 0, 0], width=R2W * 2, height=R2H * 2)


R2Mon0 = Monitor([R2XMLA - 1e-4, 0, 0], width=R2W * 3, height=R2H * 3)
R2Mon1 = Monitor([-DMMA - 1e-4, 0, 0], width=R2Wm0, height=R2Hm0)
R2blk0 = Block([R2XMLA - 5e-5, 0, 0], width=R2W * 3, height=R2H * 3)
R2blk1 = Block([-DMMA - 5e-5, 0, 0], width=R2Wm0, height=R2Hm0)

table = OpticalTable()
components = [R2m0, R2mp1, R2mp2]
components.append(R2mma if not MMAReplaceWithMir else R2m1)
if MMABlock:
    components.append(R2blk0)
if BMMABlock:
    components.append(R2blk1)

table.add_components(components)
table.add_monitors([R2Mon0, R2Mon1])
table.ray_tracing(R2rays0)


table.render(
    ax0[0],
    type=PLOT_TYPE,
    roi=[-5, R2d + 2, -R2W - 1, R2W + 1, -2, 2],
    gaussian_beam=True,
    label=False,
)
table.render(
    ax0[1],
    type="Y",
    roi=[-5, R2d + 2, -R2W - 1, R2W + 1, -2, 2],
    gaussian_beam=True,
    label=False,
    switch_axis=True,
)
table.render(
    ax1[1],
    type="Y",
    roi=[-R2Hm0, R2Hm0, -1.5, 0.2, -0.2, 0.2],
    gaussian_beam=True,
    label=False,
)

R2Mon0.render_scatter(ax1[0])
R2Mon0.render_projection(ax1[0], type="X", comp=R2mma, linewidth=0.1)
R2Mon1.render_scatter(ax1[2])
R2Mon1.render_projection(ax1[2], type="X", comp=R2m0, linewidth=0.1)


if __name__ == "__main__":
    plt.show()
