Ray
===

.. currentmodule:: src.ray

GaussianBeam
------------

.. autoclass:: GaussianBeam
   :members: q_at_waist, q_at_z, distance_to_waist, waist, rayleigh_range, radius_of_curvature, spot_size
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:

Ray
---

.. autoclass:: Ray
   :members: direction, n, transform_matrix, tangent_1, tangent_2, pathlength, phase, q_at_waist, q_at_z, distance_to_waist, waist, rayleigh_range, radius_of_curvature, spot_size, Propagate, render
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Functions
---------

.. autofunction:: multiplex_rays_in_wavelength
