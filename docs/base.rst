Base
====

.. currentmodule:: optable.base

Classes
-------

Base
----

.. autoclass:: Base
   :members: copy
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Vector
------

.. autoclass:: Vector
   :members: R, RotX, RotY, RotZ, RotXAroundLocal, RotYAroundLocal, RotZAroundLocal, TX, TY, TZ
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Path
----

.. autoclass:: Path
   :members: coord, direction, rotz_theta, bbox
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Color
-----

.. autoclass:: Color
   :member-order: bysource
   :show-inheritance:
   :no-inherited-members:
   :special-members: __init__

Functions
---------

.. autofunction:: run_code_block

.. autofunction:: to_mathematical_str

.. autofunction:: get_attr_str

.. autofunction:: base_merge_bboxs

.. autofunction:: wavelength_to_rgb
