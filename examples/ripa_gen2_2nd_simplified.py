import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy

from optable import *

# ── Tunable parameters [value, min, max] ──
ripa_2nd_demo = {
    "R2MLAroc": [3, 3, 3],
    "R2dXMLA": [0.0, -0.3, 0.4],
    "R2dY4F": [0, -0.5, 0.5],
    "R2kappa": [-1.0394, -2, 2],
    "R2a4": [-0.00038561165773536645, -0.04, 0.04],
    "R2a6": [0.006354907703229353, -0.04, 0.04],
}

presets = {"default": ripa_2nd_demo}

vars = ripa_2nd_demo
for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")

# ── Constants ──
R2R = 0.98
R2wl = 780e-7  # all physical quantity is in unit of cm, 780nm
R2DMLA = 360e-4  # 360um
R2NMLA = 100
R2MMLA = 1
R2L = R2DMLA * R2NMLA / 2
R2Dm = 2.54 * 2
R2X_waist = 0

# ── Derived quantities ──
R2w0 = np.sqrt(R2wl * R2MLAroc / (2 * np.pi))
print("R2w0=", R2w0 * 1e4, "um")
print("R2DMLA/R2w0=", R2DMLA / R2w0)
clipping_loss = np.exp(-2 * ((R2DMLA / 2) ** 2) / (R2w0**2))
print("clipping_loss=", clipping_loss)
print("NMLA*clipping_loss=", R2NMLA * clipping_loss)
R2d = R2MLAroc / 2
R2theta0 = np.arctan(R2DMLA / R2d / 2)
print("R2theta0=", R2theta0)
R2ZRays = [(i - (R2MMLA - 1) / 2) * R2DMLA for i in range(R2MMLA)]


# ── Rays (reverse ray tracing, intersecting) ──
def get_r2_rays(idxs, sign, offset=0):
    return [
        Ray(
            [0, R2L - (i + offset) * R2DMLA, R2ZRay],
            [1, 0, 0],
            wavelength=R2wl,
            w0=R2w0,
            id=int(id + R2NMLA * int(sign == -1)),
        )
        .Propagate(-R2X_waist)
        ._RotAroundLocal([0, 0, 1], [0, 0, 0], -sign * R2theta0)
        for id, i in enumerate(idxs)
        for R2ZRay in R2ZRays
    ]


SPACING = 10  # Change this to increase/decrease the number of rays
OFFSET = 1
R2rays_m = get_r2_rays(range(0, R2NMLA, SPACING), sign=1)
R2rays_p = get_r2_rays(range(0, R2NMLA, SPACING), offset=OFFSET, sign=-1)
R2rays = deepcopy(R2rays_m + R2rays_p)


# ── Lens pair (LENS=6: Aspheric Parametric, f=37.5, UVFS glass) ──
EFL = 37.5
CT = 0.8
n = Glass_UVFS()
R = EFL * (n.n(780e-9) - 1)
print("R=", R * 10, "mm")
DL = 2.54 * 3
F1 = 37.5
F2 = 37.5

R2l0r = ASphericParametricLens(
    [F1, 0, 0],
    CT=CT,
    diameter=DL,
    R=R,
    n=n,
    kappa=R2kappa,
    a4=R2a4 * (1e-3 / 1e-2) ** 4,
    a6=R2a6 * (1e-3 / 1e-2) ** 6,
    name="L0",
).TY(R2DMLA / 2 + R2dY4F)

R2l1r = (
    ASphericParametricLens(
        [F1 + 2 * F2, 0, 0],
        CT=CT,
        diameter=DL,
        R=R,
        n=n,
        kappa=R2kappa,
        a4=R2a4 * (1e-3 / 1e-2) ** 4,
        a6=R2a6 * (1e-3 / 1e-2) ** 6,
        name="L1",
    )
    .RotZ(np.pi)
    .TY(R2DMLA / 2 + R2dY4F)
)


# ── Monitors ──
R2Mon0 = Monitor([-0.5, -2.5, 0], width=5, height=5, name="R2 Monitor 0").TY(R2L)
R2Mon1 = Monitor([2 * F1 + 2 * F2, 0, 0], width=5, height=5, name="R2 Monitor 1")
R2Mon2 = Monitor(
    [2 * F1 + 2 * F2 + R2d + 0, 0, 0], width=5, height=5, name="R2 Monitor 2"
)

# ── Assemble component group ──
ripa2 = ComponentGroup([0, 0, 0], name="RIPA2")
ripa2.add_rays(R2rays)
ripa2.add_components([R2l0r, R2l1r])
ripa2.add_monitors([R2Mon0, R2Mon1, R2Mon2])

# ── Ray tracing ──
table = OpticalTable()
table.add_components([ripa2])
table.add_monitors(ripa2.monitors)
table.ray_tracing(ripa2.rays)


if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    ax1 = [plt.subplot(gs1[i]) for i in range(3)]

    table.render(
        ax0,
        type="Z",
        roi=(-5, 2 * F1 + 2 * F2 + R2d + 2, -5, 5, -1, 1),
        gaussian_beam=True,
        label=False,
        physical_color=False,
    )

    table.render(
        ax1[0],
        type="Z",
        roi=(
            2 * F1 + 2 * F2 - 0.3,
            2 * F1 + 2 * F2 + R2d + 2,
            -R2L * 2,
            R2L * 2,
        ),
        gaussian_beam=True,
    )

    table.render(
        ax1[1],
        type="Z",
        roi=(-0.2 - R2d, 2 - R2d, -1, 1),
        gaussian_beam=True,
    )

    # ── Solve ray-ray intersections ──
    R2NNMLA = len(R2rays_m)
    rays_Mon = R2Mon2.get_rays(sort="ID")
    print(len(rays_Mon))
    PList = []
    nList = []
    rocList = []
    pathLengthList = []
    for i in range(R2NNMLA):
        ray0i = rays_Mon[i]
        ray0i_id = ray0i._id
        ray1i, _ = R2Mon2.get_ray_id(ray0i_id + R2NMLA)
        t0, t1, P, n = solve_ray_ray_intersection(
            ray0i.origin, ray0i.direction, ray1i.origin, ray1i.direction
        )
        d0 = ray0i.distance_to_waist(ray0i.q_at_z(t0))
        d1 = ray1i.distance_to_waist(ray1i.q_at_z(t1))
        pl0 = ray0i.pathlength(t0)
        pl1 = ray1i.pathlength(t1)
        d = 1.5
        roci = 2 * (d**2 + d0 * d1) / (d0 + d1)
        ax0.scatter(P[0], P[1], color="magenta", s=10)
        ax1[0].scatter(P[0], P[1], color="magenta", s=10)
        # print(d1, roci, P, n)
        PList.append(P)
        nList.append(n)
        rocList.append(roci)
        pathLengthList.append((pl0 + pl1) / 2)

    stdCX = np.std([P[0] for P in PList])
    print("std of intersection points in X direction (cm)=", stdCX)

    pathlengths = np.array(pathLengthList) / R2wl
    pathlengths -= np.min(pathlengths)
    print("pathlengths (in unit of wavelength)=", pathlengths)

    print("See magenta dots: ray-ray intersection points")
    plt.show()
