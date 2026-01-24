# calibrate_4f.py
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


LENS = 6

if LENS == 4:
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

N = 3
D = 0.5
R2wl = 780e-7  # in cm
R2w0 = 61e-4  # in cm
r0 = [
    Ray([-10, i * D, 0], [1, 0, 0], wavelength=R2wl, w0=R2w0, id=int(i + N)).Propagate(
        -10
    )
    for i in np.arange(-N, N + 1)
]
# CRITERION = "M=-I"
CRITERION = "flat_field"
F1, F2 = OpticalTable.calibrate_symmetric_4f(
    lens0, r0, F10=EFL, F20=EFL, criterion=CRITERION, debugaxs=ax0
)
print(F1, F2)
OpticalTable.calibrate_symmetric_4f(
    lens0, r0, F10=F1, F20=F2, debugaxs=ax0, optimize=False, display_M=True
)
print("FK")
plt.show()
