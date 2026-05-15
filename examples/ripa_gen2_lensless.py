import numpy as np, sys, os, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

sys.path.append("../")
from optable import *

# PLOT_TYPE = "X"  # "Z" or "3D"
PLOT_TYPE = "Z"  # "Z" or "3D"
# PLOT_TYPE = "3D"  # "Z" or "3D"

# DETAILED_RENDER = True
DETAILED_RENDER = False
# SHOW_MONITOR = True
SHOW_MONITOR = False

if __name__ == "__main__":
    # plot3d
    if DETAILED_RENDER:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0])
        ax0 = [
            plt.subplot(gs0[i], projection="3d" if PLOT_TYPE == "3D" else None)
            for i in range(2)
        ]
        gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
        ax1 = [plt.subplot(gs1[i]) for i in range(3)]
    else:
        fig, ax0 = plt.subplots(
            1,
            1,
            figsize=(6, 6),
            subplot_kw={"projection": "3d" if PLOT_TYPE == "3D" else None},
        )

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
#
# Constants
BW = 6.83  # in GHZ
R1ultraR = 0.9999
R1R = 0.98
wl = 780e-7  # in cm
# DMLA = 500e-4
DMLA = 420e-4
R1NMLA = 80
R1MMLA = 15

# computation
Lrt = 3e8 / (2 * BW * 1e9) / 0.01
print("Lrt=", Lrt, "cm")
#
R1MMAshify = DMLA / (R1MMLA + 1)
print("R1MMAshify=", R1MMAshify)
R1ZRay = (R1MMLA / 2) * DMLA
print("R1ZRay=", R1ZRay)
#
R1W = DMLA * R1NMLA
R1H = DMLA * R1MMLA
#
R1d = np.sqrt((Lrt / 2) ** 2 - (R1MMAshify / 2) ** 2 - (DMLA / 2) ** 2)
print("R1d=", R1d, "cm")
R2MLAROCCOEF = 1.0
# R2MLAROCCOEF = 1.2
R1MLAroc = (Lrt) * R2MLAROCCOEF
print("R1MLAroc=", R1MLAroc, "cm")
# R1w0 = np.sqrt(wl * R1MLAroc / (2 * np.pi))
R1w0 = np.sqrt(wl * (np.sqrt((Lrt) * (R1MLAroc / 2 - Lrt / 4))) / (np.pi))
print("R1w0=", R1w0 * 1e4, "um")
#
print("DMLA/R1w0=", DMLA / R1w0)
clipping_loss = np.exp(-2 * ((DMLA / 2) ** 2) / ((np.sqrt(2) * R1w0) ** 2))
print("clipping_loss=", clipping_loss)
print("MMLA*NMLA*clipping_loss=", R1MMLA * R1NMLA * clipping_loss)
#
R1theta0 = np.arctan(DMLA / R1d / 2)
print("R1theta0=", R1theta0)
# Input physical rays
R2X_waist = 0

vec_input = np.array([R1d, -R1MMAshify / 2, DMLA / 2])
vec_input = vec_input / np.linalg.norm(vec_input)
vec_output = np.array([-R1d, -R1MMAshify / 2, DMLA / 2])
vec_output = vec_output / np.linalg.norm(vec_output)
vec_output_projxz = [vec_output[0], vec_output[2]]
vec_output_projxz /= np.linalg.norm(vec_output_projxz)

R1rays0 = [
    Ray(
        [0, R1W / 2, -R1ZRay],
        vec_input,
        wavelength=wl,
        w0=R1w0,
    ).Propagate(-(R2X_waist - (0)))
]
origin_output = np.array([0, -R1MMAshify / 2, R1ZRay])
R1DMMA = 0.1


def hit_point(t):
    p = origin_output + vec_output * t
    o_BMMA = np.array([-R1DMMA, 0, 0])
    vec2 = p - o_BMMA
    # print("vec2=", vec2)
    t2 = np.linalg.norm(vec2)
    vec2_projxz = [vec2[0], vec2[2]]
    vec2_projxz = np.array(vec2_projxz) / np.linalg.norm(vec2_projxz)
    vec_mid = (vec2_projxz + vec_output_projxz) / 2
    vec_mid /= np.linalg.norm(vec_mid)
    angle = np.arcsin(vec_mid[1])
    return t + t2, p, angle


# solve t within 0-Lrt st hit_point(t)=Lrt/2
from scipy.optimize import brentq

t_solution = brentq(lambda t: hit_point(t)[0] - Lrt / 2, 0, Lrt)
_, p, angle_solution = hit_point(t_solution)
print("t_solution=", t_solution)
print("p=", p)
print("angle_solution=", angle_solution * 180 / np.pi, "deg")

R1delta = 2 * R1w0  # cut on both sides of mirror
R1Wm0 = (R1NMLA) * DMLA + 2 * R1delta
R1Hm0 = (R1MMLA) * DMLA - 2 * R1delta
R2Wm0 = (R1NMLA) * DMLA - 2 * R1delta

R1mp1 = SquareMirror(
    [p[0], 0, p[2]], width=R1Wm0, height=0.2, reflectivity=R1ultraR
).RotY(angle_solution)
R1mp2 = SquareMirror(
    [p[0], 0, -p[2]], width=R1Wm0, height=0.2, reflectivity=R1R, transmission=1
).RotY(-angle_solution)

R1_origin_next_input_xz = [0, -R1ZRay]
R1mp2_origin_xz = [p[0], -p[2]]
R1bmma_origin_xz = [-R1DMMA, 0]
vec_bmma_2_R2mp2 = np.array(R1mp2_origin_xz) - np.array(R1bmma_origin_xz)
ripa2_rotate_angle = np.arctan(vec_bmma_2_R2mp2[1] / vec_bmma_2_R2mp2[0])
print("ripa2_rotate_angle=", ripa2_rotate_angle * 180 / np.pi, "deg")
vec_bmma_2_R2mp2 /= np.linalg.norm(vec_bmma_2_R2mp2)
ripa1_output_xz = R1mp2_origin_xz + vec_bmma_2_R2mp2 * np.linalg.norm(
    np.array(R1_origin_next_input_xz) - np.array(R1mp2_origin_xz)
)
ripa1_output_refpoint = PointObj([ripa1_output_xz[0], -DMLA / 2, ripa1_output_xz[1]])
midpoint_outputref_R1mp2 = (ripa1_output_refpoint.origin + R1mp2.origin) / 2

R1BMMAshift = -(DMLA / 2) * (R1MMLA) / (R1MMLA + 1)
R1m0 = MMA(
    origin=[-R1DMMA, 0, 0],
    N=(R1NMLA, 1),
    pitch=DMLA,
    roc=R1MLAroc,
    n=1.5,
    thickness=R1DMMA,
    reflectivity=R1ultraR,
    transmission=0,
    mma_width=R1Wm0,
    mma_height=R1Hm0,
    mma_shifty=R1BMMAshift,
    back_transmission=0,
    back_reflectivity=1,
)

R1mma = MMA(
    origin=[R1d, 0, 0],
    N=(R1NMLA, R1MMLA),
    pitch=DMLA,
    roc=R1MLAroc,
    n=1.5,
    thickness=0.1,
    reflectivity=R1ultraR,
    transmission=0.0,
    render_comp_vec=False,
    name="R1MMA",
    shifty_z=0 if not MMAShift else -R1MMAshify,
).TY(DMLA / 2 - R1MMAshify / 2)

R1m1 = SquareMirror(origin=[R1d, 0, 0], width=R1W, height=R1H)


R1Mon0 = Monitor(
    [R1d - 1e-4, 0, 0], width=R1W / 2 * 3, height=R1H / 2 * 3, render_obj=SHOW_MONITOR
)
R1Mon1 = Monitor(
    [-R1DMMA - 1e-4, 0, 0], width=R1Wm0 * 2, height=R1Hm0 * 2, render_obj=SHOW_MONITOR
)
R1blk0 = Block(
    [R1d - 5e-5, 0, 0], width=R1W / 2 * 3, height=R1H / 2 * 3, render_obj=SHOW_MONITOR
)
R1blk1 = Block(
    [-R1DMMA - 5e-5, 0, 0], width=R1Wm0, height=R1Hm0, render_obj=SHOW_MONITOR
)

components = [R1m0, R1mp1, R1mp2]
components.append(R1mma if not MMAReplaceWithMir else R1m1)
if MMABlock:
    components.append(R1blk0)
if BMMABlock:
    components.append(R1blk1)

ripa1 = ComponentGroup([0, 0, 0])
ripa1.add_components(components)
ripa1.add_monitors([R1Mon0, R1Mon1])
ripa1.add_refpoint(ripa1_output_refpoint)


# begin constructing ripa2
R12blkW = 0.5
R12blk0 = Block(midpoint_outputref_R1mp2, width=R2Wm0, height=R1W, render_obj=False).TY(
    -4 * R1delta
)

R2mma = (
    MMA(
        origin=[R1d, 0, 0],
        N=(R1NMLA, R1NMLA),
        pitch=DMLA,
        roc=R1MLAroc,
        n=1.5,
        thickness=0.1,
        reflectivity=R1ultraR,
        transmission=0.0,
        render_comp_vec=False,
        name="R2MMA",
        shifty_z=0,
    )
    # .TY(DMLA / 2)
    .TZ(-R1W / 2)
)
# R2m0 = SquareMirror(origin=[0, 0, 0], width=R1Wm0 * 2, height=R2Wm0).TZ(-R1W / 2)
R2m0 = (
    TriangularPrism(
        origin=[0, 0, 0],
        width=R1Wm0,
        height=R1Wm0 * 2,
        n1=1,
        n2=1.55,
        alpha=np.pi / 4,
        beta=np.pi / 2,
        reflectivity_1=1,
        transmission_1=0.03,
        reflectivity_2=1,
        transmission_2=0,
        reflectivity_3=0,
        transmission_3=1,
        max_interact_count_2=1e5,
        max_interact_count_3=1e5,
    )
    .RotX(np.pi / 2)
    .TZ(-R1Wm0 - R1delta)
)

R2Mon0 = Monitor(
    [R1d - 1e-4, 0, 0], width=R1W * 3, height=R1H * 3, render_obj=SHOW_MONITOR
).TZ(-R1W / 2)

components = [R2mma, R2m0]
ripa2 = ComponentGroup([0, 0, 0])
ripa2.add_components(components)
ripa2.add_monitors([R2Mon0])
ripa2.RotY((np.pi - ripa2_rotate_angle - R1theta0))
# print(ripa1_output_refpoint.origin)
ripa2._Translate(ripa1_output_refpoint.origin)
# ripa2._Translate([0, 0, -1])


table = OpticalTable()
table.add_components([ripa1, ripa2])
table.add_components([R12blk0])
table.add_monitors(ripa1.monitors)
table.add_monitors(ripa2.monitors)
table.ray_tracing(R1rays0, perfomance_limit={"max_trace_num": 1e5})

if DETAILED_RENDER:
    table.render(
        ax0[0],
        type=PLOT_TYPE,
        roi=[-5, R1d + 2, -R1W / 2 - 1, R1W / 2 + 1, -2, 2],
        gaussian_beam=True,
        label=False,
    )
    table.render(
        ax0[1],
        type="Y",
        roi=[-5, R1d + 2, -R1W / 2 - 1, R1W / 2 + 1, -2, 2],
        gaussian_beam=True,
        label=False,
        switch_axis=True,
    )
    table.render(
        ax1[1],
        type="Y",
        roi=[-R1Hm0, R1Hm0, -1.5, 0.2, -0.2, 0.2],
        gaussian_beam=True,
        label=False,
    )

    R1Mon0.render_scatter(ax1[0])
    R1Mon0.render_projection(ax1[0], type="X", comp=R1mma, linewidth=0.1)
    # R1Mon1.render_scatter(ax1[2])
    # R1Mon1.render_projection(ax1[2], type="X", comp=R1m0, linewidth=0.1)
    R2Mon0.render_scatter(ax1[2])
    R2Mon0.render_projection(ax1[2], type="X", comp=R2mma, linewidth=0.1)
    R2Mon0.render_projection(ax1[2], type="X", comp=R2m0, linewidth=0.1)
else:
    table.render(
        ax0,
        type="Y",
        roi=[-5, R1d + 2, -R1W / 2 - 1, R1W / 2 + 1, -2, 5],
        gaussian_beam=True,
        label=False,
        switch_axis=True,
    )
    ax0.set_xlabel("X (cm)")
    ax0.set_ylabel("Z (cm)")
    plt.savefig("ripa_gen2_lensless.png", dpi=600)

if __name__ == "__main__":
    plt.show()
