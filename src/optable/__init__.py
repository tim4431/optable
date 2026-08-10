"""optable — a simple ray tracing and visualization tool for freespace optics.

Distributed on PyPI as ``optical-table``; imported as ``optable``.
"""

import importlib.metadata

from .base import (
    Base,
    Vector,
    Path,
    Color,
    run_code_block,
    to_mathematical_str,
    get_attr_str,
    base_merge_bboxs,
    wavelength_to_rgb,
)
from .solver import (
    solve_ray_bboxes_intersections,
    solve_ray_ray_intersection,
    solve_normal_to_normal_rotation,
)
from .material import (
    Material,
    ConstMaterial,
    Vacuum,
    SellmeierMaterial,
    RefractiveIndex,
    plot_material_refractive_index,
    Glass_NBK7,
    Glass_UVFS,
    Glass_NSF5,
    Glass_NSF11,
    Glass_NSK2,
    Glass_NSF57,
)
from .surfaces import (
    Surface,
    Point,
    Plane,
    Circle,
    Rectangle,
    Cylinder,
    Sphere,
    ASphere,
    Polygon,
)
from .ray import (
    GaussianBeam,
    Ray,
    multiplex_rays_in_wavelength,
)
from .optical_component import (
    OpticalComponent,
    PointObj,
    Block,
    BaseMirror,
    BaseRefraciveSurface,
    Mirror,
    SquareMirror,
    SquareRefractive,
    CircleRefractive,
    SphereRefractive,
    BeamSplitter,
    Lens,
    CylMirror,
)
from .component_group import (
    ComponentGroup,
    GlassSlab,
    CircleGlassSlab,
    MLA,
    MMA,
    MMADisordered,
    DMD,
    WedgePlate,
    MirrorPair,
    Prism,
    TriangularPrism,
    MirrorPrism,
    MirrorCube,
    DovePrism,
    PlanoConvexLens,
    BiConvexLens,
    Doublet,
    ASphericLens,
    ASphericExactSphericalLens,
    ASphericParametricLens,
)
from .monitor import Monitor
from .optical_table import OpticalTable

__all__ = [
    # base
    "Base",
    "Vector",
    "Path",
    "Color",
    "run_code_block",
    "to_mathematical_str",
    "get_attr_str",
    "base_merge_bboxs",
    "wavelength_to_rgb",
    # solver
    "solve_ray_bboxes_intersections",
    "solve_ray_ray_intersection",
    "solve_normal_to_normal_rotation",
    # material
    "Material",
    "ConstMaterial",
    "Vacuum",
    "SellmeierMaterial",
    "RefractiveIndex",
    "plot_material_refractive_index",
    "Glass_NBK7",
    "Glass_UVFS",
    "Glass_NSF5",
    "Glass_NSF11",
    "Glass_NSK2",
    "Glass_NSF57",
    # surfaces
    "Surface",
    "Point",
    "Plane",
    "Circle",
    "Rectangle",
    "Cylinder",
    "Sphere",
    "ASphere",
    "Polygon",
    # ray
    "GaussianBeam",
    "Ray",
    "multiplex_rays_in_wavelength",
    # optical_component
    "OpticalComponent",
    "PointObj",
    "Block",
    "BaseMirror",
    "BaseRefraciveSurface",
    "Mirror",
    "SquareMirror",
    "SquareRefractive",
    "CircleRefractive",
    "SphereRefractive",
    "BeamSplitter",
    "Lens",
    "CylMirror",
    # component_group
    "ComponentGroup",
    "GlassSlab",
    "CircleGlassSlab",
    "MLA",
    "MMA",
    "MMADisordered",
    "DMD",
    "WedgePlate",
    "MirrorPair",
    "Prism",
    "TriangularPrism",
    "MirrorPrism",
    "MirrorCube",
    "DovePrism",
    "PlanoConvexLens",
    "BiConvexLens",
    "Doublet",
    "ASphericLens",
    "ASphericExactSphericalLens",
    "ASphericParametricLens",
    # monitor
    "Monitor",
    # optical_table
    "OpticalTable",
]

try:
    __version__ = importlib.metadata.version("optical-table")
except importlib.metadata.PackageNotFoundError:
    # Package is not installed (e.g. running from a plain checkout)
    __version__ = "unknown"
