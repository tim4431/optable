"""Interactive 1188 nm monolithic cat's eye; run this file directly.

    python examples/cat_eye_reflector.py
    python examples/cat_eye_reflector.py --save exports/cat-eye.png --no-show

Sliders move a fixed manufactured optic. Lengths in the model are metres;
plots/controls use mm, micrometres and mrad. See cat_eye_reflector.md.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from optable import Glass_UVFS, gaussian_mode_overlap


@dataclass(frozen=True)
class CatEyeDesign:
    """Spherical AR entrance, flat HR rear; rotation pivots at front vertex.

    All lengths are SI metres and angles radians, independent of scene units.
    The Gaussian model is first order; exact meridional Snell rays provide
    a geometric check, not a diffraction/aberration calculation.

    The default thickness is solved once for the nominal input mode.
    Translating the object never reoptimizes that thickness.
    """

    wavelength: float = 1188e-9
    waist: float = 200e-6
    distance: float = 50e-3
    roc: float = 5e-3
    aperture_radius: float = 1.5e-3
    index: float = None

    def __post_init__(self):
        if self.index is None:
            object.__setattr__(self, "index", float(Glass_UVFS().n(self.wavelength)))
        values = (self.wavelength, self.waist, self.distance, self.roc,
                  self.aperture_radius, self.index)
        if not np.all(np.isfinite(values)) or min(values) <= 0 or self.index <= 1:
            raise ValueError("Finite positive dimensions and index > 1 are required")
        if self.aperture_radius >= self.roc:
            raise ValueError("Clear-aperture radius must be smaller than surface ROC")
        sag = self.roc - np.sqrt(self.roc**2 - self.aperture_radius**2)
        if self.thickness <= sag:
            raise ValueError("This curvature does not give a physical rear-waist design")

    @property
    def rayleigh_range(self):
        return np.pi * self.waist**2 / self.wavelength

    @property
    def power(self):
        return (self.index - 1) / self.roc

    @property
    def thickness(self):
        q_front = self.distance + 1j * self.rayleigh_range
        return -self.index * (1 / (1 / q_front - self.power)).real

    def radius(self, reduced_q):
        """1/e^2 intensity radius; reduced_q = physical q / local index."""
        return np.sqrt(-self.wavelength / (np.pi * np.imag(1 / reduced_q)))

    def simulate(self, axial=0., decenter=0., tilt=0., thickness_error=0., samples=181):
        """Return Gaussian mode and paths for one transverse meridian.

        Positive tilt rotates the object axis from +x toward +y. Return angle
        is dy/ds with s increasing toward -x. Orthogonal alignment is identical
        by rotational symmetry; roll has no effect for this circular model.
        """
        if not np.all(np.isfinite([axial, decenter, tilt, thickness_error])):
            raise ValueError("Alignment parameters must be finite")
        length, t, n, phi = self.distance + axial, self.thickness + thickness_error, self.index, self.power
        sag = self.roc - np.sqrt(self.roc**2 - self.aperture_radius**2)
        if length <= 0 or t <= sag:
            raise ValueError("The front must follow the waist and the back must follow the cap")
        q0 = 1j * self.rayleigh_range
        qf = q0 + length
        qg = 1 / (1 / qf - phi)
        qm = qg + t / n
        qb = qm + t / n
        qr = 1 / (1 / qb - phi)
        qout = qr + length
        # Reduced ray coordinate p=n*theta, with unfolded return propagation.
        kick = phi * decenter + (n - 1) * tilt
        pf = kick
        ym = t / n * pf
        pb = pf - 2 * n * tilt
        yf = ym + t / n * pb
        pr = pb - phi * yf + kick
        y0 = yf + length * pr
        overlap = gaussian_mode_overlap(q0, qout, self.wavelength, (y0, 0), (pr, 0))
        air = np.linspace(0, length, samples)
        glass = np.linspace(0, t, samples)
        # Each path has columns x, centre y, Gaussian radius.
        incoming = np.column_stack((air, np.zeros_like(air), self.radius(q0 + air)))
        forward = np.column_stack((length + glass, pf * glass / n, self.radius(qg + glass / n)))
        backward = np.column_stack((length + t - glass, ym + pb * glass / n, self.radius(qm + glass / n)))
        returning = np.column_stack((length - air, yf + pr * air, self.radius(qr + air)))
        margin = min(self.aperture_radius - abs(decenter) - 2 * float(self.radius(qf)),
                     self.aperture_radius - abs(yf - decenter) - 2 * float(self.radius(qr)),
                     self.aperture_radius - abs(ym - decenter - tilt*t) - 2 * float(self.radius(qm)))
        return dict(overlap=overlap, q_return=qout, q_mirror=qm, displacement=y0,
                    angle=pr, return_radius=float(self.radius(qout)),
                    mirror_radius=float(self.radius(qm)), front_distance=length,
                    thickness=t, aperture_margin=margin,
                    paths=(incoming, forward, backward, returning))

    def trace_ray(self, axial=0., decenter=0., tilt=0., thickness_error=0.,
                  height=0., incident_angle=0.):
        """Exact 2D Snell/reflection ray through the moved spherical solid.

        Input ray starts at x=0, y=height. Returns five lab points (start,
        entrance, rear, exit, x=0) or None for a missed/clipped/TIR ray.
        Does not imply exact Gaussian overlap at large tilt/decenter.
        """
        c, s = np.cos(tilt), np.sin(tilt)
        rotation = np.array([[c, -s], [s, c]])
        vertex = np.array([self.distance + axial, decenter])
        start = np.array([0., height])
        p = rotation.T @ (start - vertex)
        v = rotation.T @ np.array([np.cos(incident_angle), np.sin(incident_angle)])
        center = np.array([self.roc, 0.])
        t = self.thickness + thickness_error

        def hit_cap(origin, direction):
            d = origin - center
            b = np.dot(d, direction)
            disc = b*b - (np.dot(d, d) - self.roc**2)
            if disc < 0:
                return None
            for distance in sorted((-b - np.sqrt(disc), -b + np.sqrt(disc))):
                h = origin + distance * direction
                if distance > 1e-12 and h[0] <= self.roc and abs(h[1]) <= self.aperture_radius:
                    return h
            return None

        def refract(direction, normal, ratio):
            # normal points INTO the transmitted medium.
            tangent = ratio * (direction - np.dot(direction, normal) * normal)
            normal_squared = 1 - np.dot(tangent, tangent)
            if normal_squared < 0:
                return None
            return tangent + np.sqrt(normal_squared) * normal

        front = hit_cap(p, v)
        if front is None:
            return None
        v = refract(v, (center - front) / self.roc, 1 / self.index)
        if v is None or v[0] <= 0:
            return None
        rear = front + (t - front[0]) / v[0] * v
        if rear[0] < front[0] or abs(rear[1]) > self.aperture_radius:
            return None
        v = np.array([-v[0], v[1]])
        back = hit_cap(rear, v)
        if back is None:
            return None
        v = refract(v, (back - center) / self.roc, self.index)
        if v is None:
            return None
        direction = rotation @ v
        exit_point = rotation @ back + vertex
        if direction[0] >= 0:
            return None
        end = exit_point - exit_point[0] / direction[0] * direction
        return np.array([start, rotation @ front + vertex,
                         rotation @ rear + vertex, exit_point, end])


def create_simulation():
    design = CatEyeDesign()
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 2, left=.08, right=.96, top=.89, bottom=.33,
                            hspace=.55, wspace=.3)
    layout = fig.add_subplot(grid[0, :])
    profile = fig.add_subplot(grid[1, 0])
    sensitivity = fig.add_subplot(grid[1, 1])
    fig.suptitle("Convex-plano cat's eye | 1188 nm | input waist radius 200 µm", y=.98)
    fig.text(.08, .93, f"Fused silica n={design.index:.7f}   •   ROC=5 mm   •   "
             f"center thickness={design.thickness*1e3:.6f} mm   •   clear diameter=3 mm")
    status = fig.text(.08, .265, "", va="top", fontsize=10)
    note = fig.text(.08, .025, "Gaussian overlap: paraxial, ideal AR/HR; excludes aberrations and clipping. "
                    "Dotted line: exact chief ray.\nTilt pivot: front vertex. "
                    "Return angle is measured from −x toward +y; transverse scale exaggerated.", fontsize=9)
    specs = [("decenter", "Transverse shift (µm)", -300, 300, 1e-6),
             ("axial", "Axial shift (mm)", -10, 10, 1e-3),
             ("tilt", "Tilt (mrad)", -20, 20, 1e-3),
             ("thickness_error", "Thickness error (µm)", -200, 200, 1e-6)]
    sliders = {}
    for i, (key, label, low, high, scale) in enumerate(specs):
        axis = fig.add_axes([.25, .205 - i*.039, .60, .022])
        sliders[key] = Slider(axis, label, low, high, valinit=0, valfmt="% .1f")

    def update(_=None):
        params = {key: sliders[key].val * scale for key, _, _, _, scale in specs}
        result = design.simulate(**params)
        for axis in (layout, profile, sensitivity):
            axis.clear()
        for i, path in enumerate(result["paths"]):
            x, y, w = path.T * 1e3
            color = "C0" if i < 2 else "C1"
            layout.fill_between(x, y-w, y+w, color=color, alpha=.12)
            layout.plot(x, y, color=color, lw=1.7, ls="-" if i < 2 else "--",
                        label=("Input →" if i == 0 else "← Return") if i in (0, 3) else None)
            layout.plot(x, y-w, color=color, alpha=.5, lw=.7)
            layout.plot(x, y+w, color=color, alpha=.5, lw=.7)
        # Physical solid outline, rotated rigidly about the translated vertex.
        ys = np.linspace(-design.aperture_radius, design.aperture_radius, 100)
        xs = design.roc - np.sqrt(design.roc**2 - ys**2)
        xy = np.column_stack((np.r_[xs, result["thickness"], result["thickness"], xs[0]],
                              np.r_[ys, ys[-1], ys[0], ys[0]]))
        a = params["tilt"]
        xy = xy @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        xy += [result["front_distance"], params["decenter"]]
        layout.fill(xy[:, 0]*1e3, xy[:, 1]*1e3, color="gray", alpha=.10)
        exact = design.trace_ray(**params)
        if exact is not None:
            layout.plot(exact[:, 0]*1e3, exact[:, 1]*1e3, ":", color="C3", lw=1.5,
                        label="Exact chief ray")
        layout.axvline(result["front_distance"]*1e3, color="gray", lw=.6)
        layout.axvline((result["front_distance"]+result["thickness"])*1e3, color="gray", lw=.6)
        layout.set(xlabel="Propagation coordinate x (mm)", ylabel="Transverse y (mm)",
                   ylim=(-.95, .95), xlim=(-1, 80))
        layout.set_title("Beam centers and ±w envelopes; AR curved entrance → flat HR rear", fontsize=10)
        layout.legend(loc="upper left", fontsize=8, ncol=3)
        u = np.linspace(-.9e-3, .9e-3, 500)
        w = result["return_radius"]
        profile.plot(u*1e6, np.exp(-2*(u/design.waist)**2), label="Input", color="C0")
        profile.plot(u*1e6, (design.waist/w)**2*np.exp(-2*((u-result["displacement"])/w)**2),
                     "--", label="Return", color="C1")
        profile.set(xlabel="y at original waist (µm)", ylabel="Intensity / input peak",
                    title="Equal-power beam profiles at x = 0")
        profile.legend(fontsize=8)
        tilt_values = np.linspace(-20, 20, 81)
        eta = [design.simulate(**dict(params, tilt=t*1e-3), samples=3)["overlap"] for t in tilt_values]
        sensitivity.plot(tilt_values, np.array(eta)*100, color="C1")
        sensitivity.plot(params["tilt"]*1e3, result["overlap"]*100, "o", color="C1")
        sensitivity.set(xlabel="Object tilt (mrad)", ylabel="Mode overlap (%)", ylim=(0, 103),
                        title="Tilt sweep with current shifts / thickness")
        status.set_text(f"Power mode overlap: {100*result['overlap']:.3f}%     "
                        f"Return offset: {result['displacement']*1e6:+.2f} µm     "
                        f"Return angle: {result['angle']*1e3:+.4f} mrad\n"
                        f"Return radius: {w*1e6:.2f} µm     "
                        f"HR radius: {result['mirror_radius']*1e6:.2f} µm" +
                        ("     Aperture intersects 2w envelope: clipping not modeled" if result["aperture_margin"] < 0 else ""))
        fig.canvas.draw_idle()
        return result

    for slider in sliders.values():
        slider.on_changed(update)
    update()
    # Keep widgets alive and expose the callback for noninteractive validation.
    fig.cat_eye_controls = sliders
    fig.cat_eye_update = update
    return fig, design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="Save the initial figure")
    parser.add_argument("--no-show", action="store_true", help="Run without a GUI")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    fig, design = create_simulation()
    result = design.simulate()
    print(f"n = {design.index:.9f}; ROC = {design.roc*1e3:.3f} mm; "
          f"thickness = {design.thickness*1e3:.9f} mm")
    print(f"HR waist = {result['mirror_radius']*1e6:.6f} um; "
          f"nominal paraxial overlap = {result['overlap']:.12f}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=160)
    if not args.no_show:
        plt.show()
    return fig


if __name__ == "__main__":
    main()
