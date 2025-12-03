# vipa_1st.py
# unit in cm

import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from scipy.optimize import minimize

sys.path.append("../")
from src import *

PLOT_TYPE = "Z"  # "Z" or "3D"
# PLOT_TYPE = "3D"  # "Z" or "3D"

if __name__ == "__main__":
    # plot3d
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0], projection="3d" if PLOT_TYPE == "3D" else None)
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    ax1 = [plt.subplot(gs1[i]) for i in range(3)]

demo = {
    "Dy": [0.0, -0.1, 0.1],
    "Dthetay": [0.0, -0.1, 0.1],
}

presets = {
    "default": demo,
}

vars = demo
for var, val in vars.items():
    if var not in locals():
        exec(f"{var} = {val[0]}")


def simulate(dy, dthetay, F1, F2, lens_params: dict = {}, render=False):
    N = 5
    D = 0.5
    R2wl = 780e-7  # in cm
    R2w0 = 61e-4  # in cm
    CASE = 0
    if CASE == 0:
        r0 = [
            Ray([-10, i * D, 0], [1, 0, 0], wavelength=R2wl, w0=R2w0, id=int(i + N))
            .Propagate(-10)
            .TY(dy)
            ._RotAroundLocal([0, 0, 1], [10, 0, 0], dthetay)
            for i in np.arange(-N, N + 1)
        ]
    elif CASE == 1:
        R2DMLA = 360e-4
        R2NMLA = 10
        R2L = R2DMLA * R2NMLA / 2
        R2MLAroc = 3.0
        R2d = R2MLAroc / 2
        R2theta0 = np.arctan(R2DMLA / R2d / 2)
        r0 = [
            Ray([-10, R2L, 0], [1, 0, 0], wavelength=R2wl, w0=R2w0, id=0)
            .Propagate(-10)
            .TY(dy)
            ._RotAroundLocal([0, 0, 1], [10, 0, 0], -R2theta0 + dthetay),
        ]

    # LENS = 0
    # LENS = 1
    # LENS = 2
    # LENS = 3
    # LENS = 4
    LENS = 5

    if LENS == 0:
        R2l0r = Lens([F1, 0, 0], focal_length=F2, radius=2.54)
        R2l1r = Lens([F1 + 2 * F2, 0, 0], focal_length=F2, radius=2.54)
    #
    elif LENS == 1:
        # Thorlabs LA1634-B
        EFLr = 35
        CT = 0.48
        Rr = 18.09
        DL = 2 * 2.54
        #
        R2l0r = PlanoConvexLens(
            [F1, 0, 0],
            EFL=EFLr,
            CT=CT,
            diameter=DL,
            R=Rr,
            name="L0",
        )

        R2l1r = PlanoConvexLens(
            [F1 + 2 * F2, 0, 0],
            EFL=EFLr,
            CT=CT,
            diameter=DL,
            R=Rr,
            name="L1",
        ).RotZ(np.pi)
    #
    elif LENS == 2:
        # Meniscus Lens, Thorlabs LE4984
        EFLr = 30
        R1 = 9.76
        R2 = 32.76
        CT = 0.54
        DL = 2 * 2.54
        #
        R2l0r = BiConvexLens(
            [F1, 0, 0],
            EFL=EFLr,
            CT=CT,
            R1=R1,
            R2=R2,
            diameter=DL,
            name="L0",
        )
        R2l1r = BiConvexLens(
            [F1 + 2 * F2, 0, 0],
            EFL=EFLr,
            CT=CT,
            R1=R1,
            R2=R2,
            diameter=DL,
            name="L1",
        ).RotZ(np.pi)
    #
    elif LENS == 3:
        # Thorlabs LA1353 - 75mm
        EFLr = 20
        CT = 1.01
        Rr = 10.3
        DL = 7.5
        #
        R2l0r = PlanoConvexLens(
            [F1, 0, 0],
            EFL=EFLr,
            CT=CT,
            diameter=DL,
            R=Rr,
            name="L0",
        )

        R2l1r = PlanoConvexLens(
            [F1 + 2 * F2, 0, 0],
            EFL=EFLr,
            CT=CT,
            diameter=DL,
            R=Rr,
            name="L1",
        ).RotZ(np.pi)
    #
    elif LENS == 4:
        DL = 2.54 * 3
        R1 = lens_params.get("R1", 10.0)
        R2 = lens_params.get("R2", 10.0)
        CT = lens_params.get("CT", 0.5)
        n = lens_params.get("n", 1.5)
        # best form lens biconvex
        R2l0r = BiConvexLens(
            [F1, 0, 0],
            CT=CT,
            R1=R1,
            R2=-R2,
            n=n,
            diameter=DL,
            name="L0",
        )
        R2l1r = BiConvexLens(
            [F1 + 2 * F2, 0, 0],
            CT=CT,
            R1=R1,
            R2=-R2,
            n=n,
            diameter=DL,
            name="L1",
        ).RotZ(np.pi)
    #
    elif LENS == 5:
        # Newport Doublet PAC095, EFL=25cm
        DL = 7.62
        R1 = 14.0894
        R2 = -10.172
        R3 = -45.4867
        CT1 = 1.45
        CT2 = 0.4
        n12 = 1.5112  # N-BK7
        n23 = 1.6595  # N-SF5
        #
        R2l0r = Doublet(
            [F1, 0, 0],
            CT1=CT1,
            CT2=CT2,
            R1=R1,
            R2=R2,
            R3=R3,
            n12=n12,
            n23=n23,
            diameter=DL,
            name="L0",
        )
        R2l1r = Doublet(
            [F1 + 2 * F2, 0, 0],
            CT1=CT1,
            CT2=CT2,
            R1=R1,
            R2=R2,
            R3=R3,
            n12=n12,
            n23=n23,
            diameter=DL,
            name="L1",
        ).RotZ(np.pi)

    Mon0 = Monitor([F1 + F2, 0, 0], width=5, height=5, name="Monitor 0")
    Mon1 = Monitor([2 * F1 + 2 * F2, 0, 0], width=10, height=10, name="Monitor 1")

    table = OpticalTable()
    components = [R2l0r, R2l1r]
    table.add_components(components)
    monitors = [Mon0, Mon1]
    table.add_monitors(monitors)
    table.ray_tracing(r0)
    if render:
        table.render(
            ax0,
            type=PLOT_TYPE,
            roi=(-3, 2 * F1 + 2 * F2 + 3, -5, 5, -1, 1),
            gaussian_beam=True,
        )
        Mon0.render_scatter(ax1[0], annote_delta_pos=True)
        Mon1.render_scatter(ax1[1], annote_delta_pos=True)
    yList = []
    tyList = []
    for ray in r0:
        hit_point = Mon1.get_ray_id(ray._id)[1]
        yList.append(hit_point[0])
        tyList.append(hit_point[2])
    fyList = Mon0.yList.copy()
    return yList, tyList, fyList


def calculate_abcd(F1, F2, lens_params: dict = {}, **kwargs):
    # Calculate ABCD matrix numerically for each ray in r0
    # return a list of ABCD matrix elements
    dy = 1e-5
    dthetay = 1e-5
    y0List, ty0List, _ = simulate(0, 0, F1, F2, lens_params, **kwargs)
    y1List, ty1List, _ = simulate(dy, 0, F1, F2, lens_params)
    y2List, ty2List, _ = simulate(0, dthetay, F1, F2, lens_params)
    ABCDList = []
    for i in range(len(y0List)):
        y0 = y0List[i]
        ty0 = ty0List[i]
        y1 = y1List[i]
        ty1 = ty1List[i]
        y2 = y2List[i]
        ty2 = ty2List[i]
        A = (y1 - y0) / dy
        B = (y2 - y0) / dthetay
        C = (ty1 - ty0) / dy
        D = (ty2 - ty0) / dthetay
        ABCDList.append((A, B, C, D))

    return [np.array([[ABCD[0], ABCD[1]], [ABCD[2], ABCD[3]]]) for ABCD in ABCDList]


def calculate_spherical_abbr(F1, F2, lens_params: dict = {}, **kwargs):
    _, _, fyList = simulate(0, 0, F1, F2, lens_params, **kwargs)
    return float(np.std(np.array(fyList)))


def objective_abcd(x):
    M = calculate_abcd(x[0], x[1], lens_params)[0]
    return np.linalg.norm(M - (-np.eye(2)))


def objective_abbr(x):
    # calculate the maximum aperature this lens can support
    R1 = x[0]
    R2 = x[1]
    CT = x[2]
    lens_params = {"R1": R1, "R2": R2, "CT": CT, "n": n}
    # ray at y, when CT=R1-\sqrt{R1^2-y^2} +R2-\sqrt{R2^2-y^2}, the lens edge
    ETmin = 0.3
    DY = 3 * 2.54
    if (
        (R1 <= DY / 2)
        or (R2 <= DY / 2)
        or (
            CT
            - (R1 - np.sqrt(R1**2 - (DY / 2) ** 2))
            - (R2 - np.sqrt(R2**2 - (DY / 2) ** 2))
            < ETmin
        )
    ):
        return 1
    return calculate_spherical_abbr(F0, F0, lens_params)


if __name__ == "__main__":
    TYPE = 0
    # TYPE = 1
    #
    F0 = 25.0
    n = 1.48
    Rr = 2 * F0 * (n - 1) / n
    CT = 3
    Rr1 = 13.868707196510456
    Rr2 = 82.63084526615549
    CT = 0.9214892123954423
    # lens_params = {"R1": Rr1, "R2": Rr2, "CT": CT}
    lens_params = {"R1": Rr1, "R2": Rr2, "CT": CT, "n": n}

    if TYPE == 0:
        F0 = 27.0

        F1 = F0
        F2 = F0
        F1 = 27.491759905229046
        F2 = 28.63518199145667

        #
        print(calculate_abcd(F1, F2, lens_params, render=True)[0])
        plt.show()
        #
        res = minimize(objective_abcd, [F1, F2], method="Nelder-Mead")
        #
        print(f"F1 = {res.x[0]}")
        print(f"F2 = {res.x[1]}")

    elif TYPE == 1:

        print(calculate_spherical_abbr(F0, F0, lens_params, render=True))
        plt.show()

        res = minimize(
            objective_abbr,
            [Rr1, Rr2, CT],
            bounds=[(0, None), (0, None), (0, 5)],
            method="Nelder-Mead",
            options={"disp": True, "xatol": 1e-6, "fatol": 1e-9, "maxiter": 5000},
        )
        print(f"Rr1 = {res.x[0]}")
        print(f"Rr2 = {res.x[1]}")
        print(f"CT =  {res.x[2]}")
