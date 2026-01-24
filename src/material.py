from .base import *


class Material:
    def __init__(self, name: str, n: Union[Callable, float]):
        self.name = name
        # n_func: function that takes wavelength (in m) and returns refractive index
        if isinstance(n, (int, float)):
            self.n_func = lambda wavelength: n  # constant refractive index
        else:
            self.n_func = n

    def n(self, wavelength_m: float) -> float:
        """Get refractive index at given wavelength (in m)

        Args:
            wavelength (float): wavelength in m
        Returns:
            float: refractive index
        """
        return self.n_func(wavelength_m)


def plot_material_refractive_index(
    material: Material, wl_min_m: float = 0.25e-6, wl_max_m: float = 2.5e-6
):
    """Plot refractive index over a wavelength range

    Args:
        wl_min (float): minimum wavelength in cm
        wl_max (float): maximum wavelength in cm
    """
    wl_list = np.linspace(wl_min_m, wl_max_m, 100)
    n_list = [material.n(wl) for wl in wl_list]
    plt.plot(np.array(wl_list) * 1e6, n_list)  # convert to microns
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("Refractive Index")
    plt.title(f"Refractive Index of {material.name}")
    plt.grid()
    plt.show()


class RefractiveIndex:
    def __init__(self, storage_name: str):
        # storage_name is where the actual Material object is kept (e.g., '_m1')
        self.storage_name = storage_name
        # self.storage_name = storage_name + "_rfidx"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Get the material object from the instance
        # material = getattr(instance, self.storage_name, None)
        # Use __dict__ to bypass the descriptor and avoid recursion
        material = instance.__dict__.get(self.storage_name, None)
        if material is None:
            raise AttributeError(f"Material for {self.storage_name} not initialized.")

        def get_n(wavelength_m=None):
            wl_m = (
                wavelength_m
                if wavelength_m is not None
                else instance.wavelength * instance.unit
            )
            return float(material.n(wl_m))

        return get_n

    def __set__(self, instance, value: Union[float, Material]):
        if isinstance(value, Material):
            # setattr(instance, self.storage_name, value)
            # Use __dict__ to bypass the descriptor and avoid recursion
            instance.__dict__[self.storage_name] = value
        else:
            # setattr(
            #     instance,
            #     self.storage_name,
            #     Material("Constant", n=float(value)),
            # )
            instance.__dict__[self.storage_name] = Material("Constant", n=float(value))


class Vacuum(Material):
    def __init__(self):
        super().__init__("Vacuum", n=1.0)


class SellmeierMaterial(Material):
    def __init__(self, name: str, Bs: List[float], Cs: List[float]):
        """Sellmeier material model

        Args:
            name (str): material name
            B (List[float]): Sellmeier B coefficients
            C (List[float]): Sellmeier C coefficients (in microns^2)
        """
        self.Bs = Bs
        self.Cs = Cs
        super().__init__(name, self.sellmeier_n)

    def sellmeier_n(self, wavelength_m: float) -> float:
        """Calculate refractive index using Sellmeier equation

        Args:
            wavelength (float): wavelength in cm

        Returns:
            float: refractive index
        """
        wl_um = wavelength_m / (1e-6)  # convert to microns
        wl_um2 = wl_um**2
        n2 = 1.0
        for Bi, Ci in zip(self.Bs, self.Cs):
            n2 += Bi * wl_um2 / (wl_um2 - Ci)
        return np.sqrt(n2)


class Glass_NBK7(SellmeierMaterial):
    def __init__(self):
        # NBK7 Sellmeier coefficients
        Bs = [1.03961212, 0.231792344, 1.01046945]
        Cs = [0.00600069867, 0.0200179144, 103.560653]
        super().__init__("BK7", Bs, Cs)


class Glass_UVFS(SellmeierMaterial):
    def __init__(self):
        # UV Fused Silica Sellmeier coefficients
        Bs = [0.6961663, 0.4079426, 0.8974794]
        Cs = [0.0684043**2, 0.1162414**2, 9.896161**2]
        super().__init__("UV Fused Silica", Bs, Cs)
