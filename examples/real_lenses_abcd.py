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


def simulate(dy, dthetay, F1, F2, F3, lens_params: dict = {}, render=False, N: int = 0):
    # print(N)
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

    # LENS = 0  # Ideal lens
    # LENS = 1  # F=35 Plano-Convex
    # LENS = 2  # F=30 Meniscus
    # LENS = 3  # F=20 Plano-Convex
    # LENS = 4 # Biconvex best form
    # LENS = 5 # Doublet
    # LENS = 6  # ASpheric
    # LENS = 7  # Meniscus+ BiConvex+Meniscus
    # LENS = 8  # Custom lens
    LENS = 9  # Jon's proposal

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
        ).RotZ(np.pi)

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
    #
    elif LENS == 6:
        EFL = 20
        CT = 1
        DL = 2.54 * 3
        R2l0r = ASphericExactSphericalLens(
            [F1, 0, 0], EFL=EFL, CT=CT, diameter=DL, n=1.5
        ).RotZ(np.pi)
        R2l1r = ASphericExactSphericalLens(
            [F1 + 2 * F2, 0, 0], EFL=EFL, CT=CT, diameter=DL, n=1.5
        )
    #
    elif LENS == 7:
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
        R2l2r = BiConvexLens([F1 + F2, 0, 0], CT=1, R1=30, R2=-30, diameter=DL, EFL=30)
    #
    elif LENS == 8:
        DL = 2.54 * 3
        R11 = lens_params.get("R11", 10.0)
        R21 = lens_params.get("R21", 10.0)
        CT1 = lens_params.get("CT1", 0.5)
        R12 = lens_params.get("R12", 10.0)
        R22 = lens_params.get("R22", 10.0)
        CT2 = lens_params.get("CT2", 0.5)
        n = lens_params.get("n", 1.5)
        # best form lens biconvex
        R2l0r = BiConvexLens(
            [F1, 0, 0],
            CT=CT1,
            R1=R11,
            R2=R21,
            n=n,
            diameter=DL,
            name="L0",
        )
        R2l1r = BiConvexLens(
            [F1 + 2 * F2, 0, 0],
            CT=CT2,
            R1=R12,
            R2=R22,
            n=n,
            diameter=DL,
            name="L1",
        ).RotZ(np.pi)
    elif LENS == 9:
        DL = 2.54 * 3
        # F0 = 25
        Rr = 10
        R2l0r = PlanoConvexLens(
            [2 * F0, 0, 0], EFL=F0, CT=1, diameter=DL, R=Rr, name="L0"
        ).RotZ(np.pi)
        R2l1r = BiConvexLens(
            [3 * F0, 0, 0], EFL=-F0 / 2, CT=0.5, R1=-Rr, R2=Rr, diameter=DL, name="L1"
        )
        R2l2r = PlanoConvexLens(
            [4 * F0, 0, 0], EFL=F0, CT=1, diameter=DL, R=Rr, name="L2"
        )

    Mon0 = Monitor([F1 + F2, 0, 0], width=5, height=5, name="Monitor 0")
    Mon1 = Monitor([F1 + 2 * F2 + F3, 0, 0], width=10, height=10, name="Monitor 1")

    table = OpticalTable()
    components = [R2l0r, R2l1r]
    if LENS == 7 or LENS == 9:
        components.append(R2l2r)
    table.add_components(components)
    monitors = [Mon0, Mon1]
    table.add_monitors(monitors)
    table.ray_tracing(r0)
    if render:
        table.render(
            ax0,
            type=PLOT_TYPE,
            roi=(-5, F1 + 2 * F2 + F3 + 5, -5, 5, -1, 1),
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


def calculate_abcd(F1, F2, F3, lens_params: dict = {}, N: int = 0, **kwargs):
    # Calculate ABCD matrix numerically for each ray in r0
    # return a list of ABCD matrix elements
    dy = 1e-5
    dthetay = 1e-5
    y0List, ty0List, _ = simulate(0, 0, F1, F2, F3, lens_params, N=N, **kwargs)
    y1List, ty1List, _ = simulate(dy, 0, F1, F2, F3, lens_params, N=N)
    y2List, ty2List, _ = simulate(0, dthetay, F1, F2, F3, lens_params, N=N)
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


def calculate_4f(F1, F2, F3, lens_params: dict = {}, N: int = 0, **kwargs):
    y0List, ty0List, _ = simulate(0, 0, F1, F2, F3, lens_params, N=N, **kwargs)
    D = 0.5
    # sort with yList
    args = np.argsort(y0List)
    y0List = [y0List[i] for i in args]
    ty0List = [ty0List[i] for i in args]
    yexpectedList = np.arange(-N, N + 1) * D
    # sum (y0List-yexpectedList)^2
    # sum ty0List^2
    error_y = np.linalg.norm(np.array(y0List) - np.array(yexpectedList))
    error_ty = np.linalg.norm(np.array(ty0List))
    return error_y, error_ty


def calculate_spherical_abbr(F1, F2, F3, lens_params: dict = {}, N: int = 0, **kwargs):
    _, _, fyList = simulate(0, 0, F1, F2, F3, lens_params, N=N, **kwargs)
    return float(np.std(np.array(fyList)))


def objective_abcd_sym(x):
    M = calculate_abcd(x[0], x[1], x[0], lens_params)[0]
    return np.linalg.norm(M - (-np.eye(2)))


def objective_abcd(x):
    M = calculate_abcd(x[0], x[1], x[2], lens_params)[0]
    return np.linalg.norm(M - (-np.eye(2)))


def objective_abcd_lensx(x):
    F1 = x[0]
    F2 = x[1]
    F3 = 4 * F0 - F1 - 2 * F2
    R11 = 1 / x[2]
    R21 = 1 / x[3]
    CT1 = x[4]
    R12 = 1 / x[5]
    R22 = 1 / x[6]
    CT2 = x[7]
    lens_params = {
        "R11": R11,
        "R21": R21,
        "CT1": CT1,
        "R12": R12,
        "R22": R22,
        "CT2": CT2,
        "n": n,
    }
    if np.abs(F1 - F0) > 10 or np.abs(F2 - F0) > 10:
        return 1e6
    if not valid_lens_params(R11, R21, CT1) or not valid_lens_params(R12, R22, CT2):
        return 1e6
    error_y, error_ty = calculate_4f(F1, F2, F3, lens_params, N=3)
    error_tot_1 = 1000 * error_y + 100 * error_ty
    MList = calculate_abcd(F1, F2, F3, lens_params, N=3)
    # errors = [np.linalg.norm(M - (-np.eye(2))) for M in MList]
    # ideally M^(-1)*diag(1,-1)*M = diag(1,-1)
    errors = [
        np.linalg.norm(
            np.linalg.inv(M) @ np.array([[1, 0], [0, -1]]) @ M
            - np.array([[1, 0], [0, -1]])
        )
        for M in MList
    ]
    error_tot_2 = sum(errors)
    print(error_tot_1, error_tot_2)
    return error_tot_1 + error_tot_2


def valid_lens_params(R1, R2, CT):
    # discuss when R1,R2 are positive or negative
    # R>0 means facing incoming ray
    ETmin = 0.3
    DY = 3 * 2.54
    if np.abs(R1) < DY / 2 or np.abs(R2) < DY / 2:
        return False
    d1 = np.sign(R1) * (R1 - np.sqrt(R1**2 - (DY / 2) ** 2))
    d2 = np.sign(R2) * (R2 - np.sqrt(R2**2 - (DY / 2) ** 2))
    if (CT - d1 - d2) < ETmin:
        return False
    return True


def objective_abbr(x):
    # calculate the maximum aperature this lens can support
    R1 = x[0]
    R2 = x[1]
    CT = x[2]
    lens_params = {"R1": R1, "R2": R2, "CT": CT, "n": n}
    # ray at y, when CT=R1-\sqrt{R1^2-y^2} +R2-\sqrt{R2^2-y^2}, the lens edge
    ETmin = 0.3
    DY = 3 * 2.54
    if not valid_lens_params(R1, R2, CT):
        return 1
    return calculate_spherical_abbr(F0, F0, F0, lens_params)


if __name__ == "__main__":
    # TYPE = 0
    # TYPE = 1
    TYPE = 2
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
        F0 = 20.0

        F1 = F0
        F2 = F0
        F3 = F0
        F1 = 20.33333391820526
        F2 = 19.99990162537117

        F3 = F1
        #
        print(calculate_abcd(F1, F2, F3, lens_params, render=True, N=5)[0])
        plt.show()
        #
        # res = minimize(objective_abcd, [F1, F2, F3], method="Nelder-Mead")
        res = minimize(objective_abcd_sym, [F1, F2], method="Nelder-Mead")
        # #
        print(f"F1 = {res.x[0]}")
        print(f"F2 = {res.x[1]}")

    elif TYPE == 1:

        print(calculate_spherical_abbr(F0, F0, F0, lens_params, render=True))
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

    elif TYPE == 2:
        F0 = 25.0
        n = 1.48
        Rr = 2 * F0 * (n - 1) / n
        CT = 3
        F1 = F0
        F2 = F0
        R11 = Rr
        R21 = 100
        CT1 = CT
        R12 = Rr
        R22 = 100
        CT2 = CT

        # optimizing M=-I
        F1 = 27.223636101007486
        F2 = 25.222908926749696
        R11 = 11.499339799244467
        R21 = 262.8719789459048
        CT1 = 0.9771276603541308
        R12 = 10.795103304976625
        R22 = 101.56056898692724
        CT2 = 1.326529400727428
        # optimizing for M^-1*diag(1,-1)*M=diag(1,-1)
        F1 = 26.824294939589155
        F2 = 25.321900643152006
        R11 = 11.469668624809847
        R21 = 260.5926833313889
        CT1 = 2.8015362995577378
        R12 = 10.807376728384568
        R22 = 107.99588606889851
        CT2 = 1.0610841435366698

        # F3 = 4 * F0 - F1 - 2 * F2
        F3 = 6 * F0 - F1 - 2 * F2
        lens_params = {
            "R11": R11,
            "R21": R21,
            "CT1": CT1,
            "R12": R12,
            "R22": R22,
            "CT2": CT2,
            "n": n,
        }
        print(calculate_abcd(F1, F2, F3, lens_params, render=True, N=3)[0])
        plt.show()
        #
        res = minimize(
            objective_abcd_lensx,
            [F1, F2, 1 / R11, 1 / R21, CT1, 1 / R12, 1 / R22, CT2],
            method="Nelder-Mead",
        )
        #
        print(f"F1 = {res.x[0]}")
        print(f"F2 = {res.x[1]}")
        print(f"R11 = {1 / res.x[2]}")
        print(f"R21 = {1 / res.x[3]}")
        print(f"CT1 = {res.x[4]}")
        print(f"R12 = {1 / res.x[5]}")
        print(f"R22 = {1 / res.x[6]}")
        print(f"CT2 = {res.x[7]}")
