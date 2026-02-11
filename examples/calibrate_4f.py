# calibrate_4f.py
# unit in cm

import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from optable import *

PLOT_TYPE = "Z"  # "Z" or "3D"
# PLOT_TYPE = "3D"  # "Z" or "3D"

if __name__ == "__main__":
    # plot3d
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0], projection="3d" if PLOT_TYPE == "3D" else None)
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])
    ax1 = [plt.subplot(gs1[i]) for i in range(3)]


LENS = 8
if LENS == 2:
    # Thorlabs LA1353 - 75mm
    EFLr = 20
    CT = 1.01
    Rr = 10.3
    DL = 7.5
    F1 = 23.1577
    F2 = 19.8595
    lens0 = PlanoConvexLens(
        [F1, 0, 0],
        EFL=EFLr,
        CT=CT,
        diameter=DL,
        R=Rr,
        name="L0",
    )

elif LENS == 4:
    EFL = 20.0
    CT = 1
    n = 1.5
    R = EFL * (n - 1)
    DL = 2.54 * 3
    kappa = -1.0

    lens0 = ASphericParametricLens(
        [0, 0, 0],
        CT=CT,
        diameter=DL,
        R=R,
        n=n,
        kappa=kappa,
        name="L0",
    )
elif LENS == 5:
    # Aspheric, Thorlabs AL100200-B
    EFL = 20.0
    R = 10.224
    n = 1.517
    kappa = -1
    # a4 = 4.9646003e-08 * 1e3
    # a6 = 7.4017872e-13 * 1e3
    # a8 = 9.4141703e-18 * 1e3
    a4 = 0
    a6 = 0
    a8 = 0
    #
    CT = 1.9
    DL = 10.0
    lens0 = ASphericParametricLens(
        [0, 0, 0],
        CT=CT,
        diameter=DL,
        R=R,
        n=n,
        kappa=kappa,
        a4=a4,
        a6=a6,
        a8=a8,
        name="L0",
    )

elif LENS == 6:
    # Parametric Aspheric Lens
    EFL = 37.5
    CT = 0.8
    n = 1.5
    R = EFL * (n - 1)
    DL = 2.54 * 3
    # DXMON2 = 1.009
    F1 = 37.5
    F2 = 37.5
    lens0 = ASphericParametricLens(
        [F1, 0, 0],
        CT=CT,
        diameter=DL,
        R=R,
        n=n,
        kappa=-1,
        name="L0",
    )
elif LENS == 7:
    # Thorlabs ACT508-200-B, doublet
    EFL = 20.0
    CT1 = 0.85
    CT2 = 0.50
    R1 = 12.1
    R2 = -11.87
    R3 = -42.24
    # F1 = 21.1798
    F1 = 26.4975
    F2 = 20.6390
    lens0 = Doublet(
        [F1, 0, 0],
        CT1=CT1,
        CT2=CT2,
        R1=R1,
        R2=R2,
        R3=R3,
        n12=Glass_NSK2(),
        n23=Glass_NSF57(),
        diameter=2.54 * 2,
        name="L0",
    )
elif LENS == 8:
    # Edmund Optics, #88-597, 75mm dia, 300mm EFL, doublet
    EFL = 30.0
    CT1 = 1.359
    CT2 = 0.6
    R1 = 18.405
    R2 = -13.734
    R3 = -39.933
    F1 = 30.3964
    F2 = 31.0822
    lens0 = Doublet(
        [F1, 0, 0],
        CT1=CT1,
        CT2=CT2,
        R1=R1,
        R2=R2,
        R3=R3,
        n12=Glass_NBK7(),
        n23=Glass_NSF5(),
        diameter=7.5,
        name="L0",
    )


if __name__ == "__main__":
    N = 3
    D = 0.6
    R2wl = 780e-7  # in cm
    R2w0 = 61e-4  # in cm
    r0 = [
        Ray(
            [-10, i * D, 0], [1, 0, 0], wavelength=R2wl, w0=R2w0, id=int(i + N)
        ).Propagate(-10)
        for i in np.arange(-N, N + 1)
    ]
    # CRITERION = "M=-I"
    # CRITERION = "flat_field"
    CRITERION = "min_stdtY"
    Ms, yList, tYList = OpticalTable.calibrate_symmetric_4f(
        lens0, r0, F10=F1, F20=F2, debugaxs=ax0, optimize=False, display_M=True
    )
    idealyList = np.array([i * D for i in np.arange(-N, N + 1)])
    yList = np.array(yList)
    dYList = yList - idealyList
    print(dYList)

    plt.show()
    # exit(0)
    F1, F2 = OpticalTable.calibrate_symmetric_4f(
        lens0, r0, F10=F1, F20=F2, criterion=CRITERION, debugaxs=ax0
    )
    print(F1, F2)
    Ms, yList, tYList = OpticalTable.calibrate_symmetric_4f(
        lens0, r0, F10=F1, F20=F2, debugaxs=ax0, optimize=False, display_M=True
    )

    print("FK")
    plt.show()
