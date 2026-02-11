Optable Documentation
=====================

.. figure:: ./assets/logo.png
   :width: 140px
   :align: right
   :height: 140px
   :alt: Optable logo
   :figclass: align-right

Optable is a Python toolkit for free-space optics ray tracing and visualization.
It is designed for fast experiment modeling with composable optical components,
Gaussian beam support, and convenient plotting in 2D and 3D.

What You Can Do
===============

- Build optical setups from mirrors, lenses, refractive surfaces, beam splitters, and grouped assemblies.
- Trace ray bundles through the setup and inspect interactions at each component.
- Work with Gaussian beam parameters through ``q``-parameter propagation.
- Render full scenes and monitor outputs for alignment and optimization workflows.

Quick Example
=============

.. code-block:: python

   from optable.optical_component import Mirror, Lens
   from optable.ray import Ray
   from optable.optical_table import OpticalTable

   rays = [Ray([-10, 0, 0], [1, 0, 0], wavelength=780e-9, w0=10e-6)]
   components = [
       Mirror([0, 0, 0]).RotZ(0.2),
       Lens([4, 0, 0], focal_length=5.0, radius=0.8),
   ]

   table = OpticalTable()
   table.add_components(components)
   table.ray_tracing(rays)

Start Here
==========

.. toctree::
   :maxdepth: 2

   installation
   examples/index
   api_doc

Indices
=======

- :ref:`genindex`
- :ref:`py-modindex`
- :ref:`search`
