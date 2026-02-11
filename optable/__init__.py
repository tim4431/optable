from .base import *
from .component_group import *
from .material import *
from .monitor import *
from .optical_component import *
from .optical_table import *
from .ray import *
from .solver import *
from .surfaces import *

import importlib.metadata

try:
    __version__ = importlib.metadata.version("optable")
except importlib.metadata.PackageNotFoundError:
    # Case where package is not installed (e.g. during development)
    __version__ = "unknown"
