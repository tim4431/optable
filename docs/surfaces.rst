Surfaces
========

.. currentmodule:: src.surfaces

Classes
-------

Surface
-------

.. autoclass:: Surface
   :members: f, normal, within_boundary, parametric_boundary_3d, merge_bbox, merge_bboxs, solve_crosssection_ray_bbox_local
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Point
-----

.. autoclass:: Point
   :members: f, normal, within_boundary, parametric_boundary, get_bbox_local
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Plane
-----

.. autoclass:: Plane
   :members: f, normal, union, subtract
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Circle
------

.. autoclass:: Circle
   :members: within_boundary, parametric_boundary, get_bbox_local
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Rectangle
---------

.. autoclass:: Rectangle
   :members: within_boundary, parametric_boundary, get_bbox_local
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Cylinder
--------

.. autoclass:: Cylinder
   :members: f, normal, within_boundary, parametric_boundary, get_bbox_local
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Sphere
------

.. autoclass:: Sphere
   :members: f, normal, within_boundary, parametric_boundary, get_bbox_local
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

ASphere
-------

.. autoclass:: ASphere
   :members: roc_r, roc, f, normal, within_boundary, parametric_boundary, get_bbox_local
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Polygon
-------

.. autoclass:: Polygon
   :members: f, within_boundary, parametric_boundary, get_bbox_local
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__
