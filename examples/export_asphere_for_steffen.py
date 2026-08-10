"""Export the aspheric surface prescription (sag table + coefficients) for the
optic vendor (Steffen).

Run from this directory:  python export_asphere_for_steffen.py
Outputs land in ../exports/ :
    asphere_sag_table.csv     radius_mm, sag_mm   (vertex at r=0)
    asphere_spec.txt          human-readable prescription with units
    asphere_sag_profile.png   quick visual

The surface is the lens L0 of ripa_gen2_2nd_simplified.py.  Sag is taken
directly from the simulated lens object so it is exactly the surface the ray
trace uses (coefficients include the (1e-2/1e-3)**4 / **6 conversions).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from optable import *

# ── Design parameters (identical to ripa_gen2_2nd_simplified.py) ──
EFL = 43.17  # cm
CT = 0.8  # cm, center thickness
n = Glass_UVFS()
nval = n.n(780e-9)
R_cm = EFL * (nval - 1)  # cm, radius of curvature of the aspheric face
KAPPA = -1.01705  # conic constant (dimensionless)
A4_typed = -4.00e-10
A6_typed = 3.180e-14

# internal (cm) coefficients actually used by the traced surface
a4_int = A4_typed * (1e-2 / 1e-3) ** 3
a6_int = A6_typed * (1e-2 / 1e-3) ** 5

DL = 2.5 * 3  # cm, lens diameter (Ø70 mm); semi-aperture 38.1 mm

# ── Build the lens and pull its exact sag closure ──
lens = ASphericParametricLens(
    [EFL, 0, 0],
    CT=CT,
    diameter=DL,
    R=R_cm,
    n=n,
    kappa=KAPPA,
    a4=a4_int,
    a6=a6_int,
    name="L0",
)

# ── Convert to manufacturer units (mm) ──
R_mm = R_cm * 10.0
A4_mm = a4_int / 1e3  # mm^-3   (consistent with the simulated surface)
A6_mm = a6_int / 1e5  # mm^-5
SEMI_MM = DL / 2 * 10.0  # 38.1 mm

# ── Sag table: radius (mm) vs sag (mm), vertex at r=0 ──
r_mm = np.arange(0.0, SEMI_MM, 0.5)
r_mm = np.append(r_mm, SEMI_MM)  # include the exact edge
sag_mm = lens.f_asphere_1(r_mm / 10.0) * 10.0  # cm -> mm

# self-check: sag table must equal the standard even-asphere equation
sqrt_term = np.sqrt(1 - (1 + KAPPA) * (r_mm**2) / R_mm**2)
sag_eq = (r_mm**2 / R_mm) / (1 + sqrt_term) + A4_mm * r_mm**4 + A6_mm * r_mm**6
assert (
    np.max(np.abs(sag_mm - sag_eq)) < 1e-9
), "sag table does not match even-asphere eq!"

# ── Write outputs ──
outdir = os.path.join("..", "exports")
os.makedirs(outdir, exist_ok=True)

# CSV
csv_path = os.path.join(outdir, "asphere_sag_table.csv")
with open(csv_path, "w", newline="") as fh:
    fh.write("radius_mm,sag_mm\n")
    for r, z in zip(r_mm, sag_mm):
        fh.write(f"{r:.4f},{z:.6f}\n")

# Spec sheet
spec = f"""ASPHERIC SURFACE PRESCRIPTION  (lens L0)
============================================================
Material        : fused silica (UV-grade, UVFS / Corning 7980 equiv.)
Design wavelength: 780 nm
Refractive index : n(780 nm) = {nval:.6f}
Clear aperture   : dia {2*SEMI_MM:.1f} mm  (semi-aperture {SEMI_MM:.1f} mm)
Center thickness : {CT*10:.2f} mm  (plano-convex; 2nd face is flat)

SURFACE: even asphere, both axes in MILLIMETRES
  sag z(r) = (r^2 / R) / (1 + sqrt(1 - (1+k) * r^2 / R^2))
             + A4*r^4 + A6*r^6

  r  = radial distance from optical axis              [mm]
  z  = sag, surface height from the vertex (z=0 at r=0) [mm]

  Radius of curvature   R  = {R_mm:.4f} mm   (curvature 1/R = {1/R_mm:.6e} 1/mm)
  Conic constant        k  = {KAPPA:.5f}     (dimensionless)
  4th-order coeff       A4 = {A4_mm:.6e}  mm^-3
  6th-order coeff       A6 = {A6_mm:.6e}  mm^-5
  (8th-order and higher = 0)

Aspheric departure from the base conic at the edge (r={SEMI_MM:.1f} mm):
  A4*r^4 + A6*r^6 = {(A4_mm*SEMI_MM**4 + A6_mm*SEMI_MM**6)*1e3:.4f} um
Total sag at the edge: {sag_mm[-1]:.4f} mm

Numerical sag table: see asphere_sag_table.csv  (radius_mm, sag_mm).
Both columns are in millimetres; vertex at r=0.
"""
spec_path = os.path.join(outdir, "asphere_spec.txt")
with open(spec_path, "w", encoding="utf-8") as fh:
    fh.write(spec)

# Plot
fig, (axA, axB) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
axA.plot(r_mm, sag_mm)
axA.set_ylabel("Sag z (mm)")
axA.set_title("Aspheric surface sag (vertex at r=0)")
axA.grid(alpha=0.3)
base = (r_mm**2 / R_mm) / (1 + sqrt_term)  # conic only
axB.plot(r_mm, (sag_mm - base) * 1e3, color="red")
axB.set_ylabel("Asphere − conic (µm)")
axB.set_xlabel("Radius r (mm)")
axB.grid(alpha=0.3)
plt.tight_layout()
png_path = os.path.join(outdir, "asphere_sag_profile.png")
plt.savefig(png_path, dpi=200)

print(spec)
print("wrote:", os.path.abspath(csv_path))
print("wrote:", os.path.abspath(spec_path))
print("wrote:", os.path.abspath(png_path))
print(f"sag table rows: {len(r_mm)}  (r = 0 .. {SEMI_MM:.1f} mm, 0.5 mm step + edge)")
