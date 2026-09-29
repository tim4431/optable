# Monolithic convex-plano cat's eye

Run `python examples/cat_eye_reflector.py` from the repository root. The four
sliders move a fixed optic sideways, axially, and in tilt, or perturb its
manufactured thickness. The plots show the Gaussian envelopes, an exact
Snell-law chief ray, the returned intensity profile, and a tilt sensitivity
sweep. `CatEyeDesign` is defined locally in this example. The reusable
`gaussian_mode_overlap` function lives in `optable.ray` and is also available
as `from optable import gaussian_mode_overlap`.

A noninteractive image can be generated with:

```powershell
python examples/cat_eye_reflector.py --no-show --save exports/cat-eye.png
```

## A concrete design

The specified 200 um waist is interpreted as the **1/e² intensity radius**
of a circular, diffraction-limited Gaussian beam (M² = 1). The distance is
measured from that waist to the curved entrance **vertex**. Air index is 1.
The wavelength, geometry, and Gaussian calculations all use SI metres.

| Quantity | Nominal value |
| --- | ---: |
| Vacuum wavelength | 1188 nm |
| Input waist radius | 200 um |
| Waist to entrance vertex | 50.000 mm |
| Substrate | Fused silica |
| Substrate index at 1188 nm | 1.448186847 |
| Spherical front radius of curvature | +5.000 mm |
| Center thickness, front vertex to flat HR rear | 16.707432 mm |
| Suggested clear aperture diameter | 3.000 mm |
| Beam radius at the entrance | 221.218 um |
| Beam waist radius at HR surface | 19.8006 um |
| Input Rayleigh range | 105.778 mm |
| Nominal ideal paraxial power mode overlap | 100% |

This is a **custom starting prescription**, not a uniquely determined or
catalog-qualified part. The front ROC is a chosen degree of freedom. A
different ROC, glass, or desired thickness requires solving the condition
below again. The shape is a thick cylindrical solid with a convex entrance
cap and a plane rear. The front coating should be AR at 1188 nm and the rear
coating HR at 1188 nm **for incidence from within fused silica**. Actual
coating transmission, reflectance, phase, and polarization are not specified
or simulated. If front power transmission is T and rear power reflectance
is H, coupled power/input power is approximately `T² H eta`, excluding
absorption and aperture losses.

The fused-silica index uses the existing `Glass_UVFS` Sellmeier implementation.
Its coefficients correspond to the Malitson dispersion data:
[Malitson, JOSA 55, 1205–1209 (1965)](https://doi.org/10.1364/JOSA.55.001205).
For the paraxial cat's-eye framework, see
[Snyder, Applied Optics 14, 1825–1828 (1975)](https://doi.org/10.1364/AO.14.001825).

## Why this thickness

Use reduced Gaussian parameter `Q = q/n` and reduced ray angle `p = n theta`.
At the entrance in air:

```text
z_R = pi w0² / lambda
q_in = L + i z_R
Phi = (n - 1) / R
Q_after_front = 1 / (1/q_in - Phi)
Q_HR = Q_after_front + t/n
t = -n Re(Q_after_front)
```

The flat rear mirror lies at a Gaussian waist: `Re(Q_HR) = 0`. Its wavefront
is flat, so reflection retraces the mode. In unfolded propagation, both
passes through the front have matrix `S = [[1,0],[-Phi,1]]`, propagation in
air is `P(L) = [[1,L],[0,1]]`, and the full return matrix to the original
waist is `P(L) S P(2t/n) S P(L)`. The returned parameter equals `i z_R`.
The target is the backward-propagating input mode at the same plane, not
the forward mode with an inconsistent curvature sign.

Putting the HR at the geometric focal plane instead would give
`t = n/Phi = 16.156061 mm`. For this finite-divergence input, that gives
only **88.12%** aligned Gaussian mode overlap. Merely HR-coating the rear
of an arbitrary conventional plano-convex lens does not meet the condition.

## What rigid-body motion does

Positive tilt rotates the object axis from +x toward +y, about its front
vertex **after translation**. Return angle is `dy/ds`, with s increasing
back toward -x. The original source/waist remains fixed in the lab. The
second transverse direction behaves identically; rotation about the optic
axis has no effect for this circular, polarization-free model.

For decenter d, tilt alpha, `b=t/n`, and `g=1-b Phi`, the first-order return
position and angle just outside the front are:

```text
y_front_return = 2 b (Phi d - alpha)
theta_return = 2 g (Phi d - alpha)
y_at_original_waist = y_front_return + L theta_return
```

The following perturbations are applied one at a time to the nominal part:

| Perturbation | Return offset at input waist | Return angle | Power mode overlap |
| --- | ---: | ---: | ---: |
| +100 um transverse shift | +176.23 um | -0.6118 mrad | 41.43% |
| +1 mrad front-vertex tilt | -19.66 um | +0.06826 mrad | 98.91% |
| +10 mrad front-vertex tilt | -196.61 um | +0.6826 mrad | 33.40% |
| +5 mm axial shift | 0 | 0 | 99.875% |
| +100 um thickness error | 0 | 0 | 99.558% |

The 90% overlap limits for isolated decenter and isolated front-vertex tilt
are approximately **±34.6 um** and **±3.10 mrad**, respectively. A large
aperture prevents clipping but does not prevent loss of coherent mode overlap.
Transverse shift and tilt can compensate to first order: `d = alpha/Phi`,
or +11.156 um decenter per +1 mrad tilt. The pivot choice therefore matters.
Axial motion primarily changes q (size and curvature), and also changes
optical phase; piston phase is intentionally absent from power overlap.

Overlap is the normalized 2D complex-field integral `|integral u_target* u_return dA|²`,
including beam size, curvature, displacement, angle, and their cross terms.
The intensity-profile plot alone cannot show angle or curvature mismatch.

## Accuracy and validation

The reported overlap is **paraxial and aberration-free**, not a prediction
of perfect experimental coupling. The spherical boundary is used exactly
for the separate chief-ray trace, but spherical aberration, coma,
astigmatism, diffraction at the aperture, coating phase, and polarization
are not folded into the Gaussian overlap. The chief ray is not an intensity
envelope. At larger slider excursions, use the results as first-order
alignment estimates; detailed fabrication design needs wave-optics or
aberration-aware optimization. The 2w aperture check flags approaching
clipping without claiming to calculate the lost power.

Tests check the rear-waist and round-trip conditions, known displacement/
tilt overlap limits, numerical integration of mismatched complex Gaussian
fields, geometric focal-plane retroreflection, rigid-body symmetry, and
agreement with independent exact Snell/reflection rays at small offsets.

```powershell
python -m pytest tests/test_cat_eye.py -q
```
