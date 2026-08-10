"""Package-level contract: version metadata and a clean public namespace."""

import importlib.metadata

import optable


def test_version_matches_distribution_metadata():
    assert optable.__version__ == importlib.metadata.version("optical-table")


def test_all_has_no_duplicates():
    assert len(optable.__all__) == len(set(optable.__all__))


def test_all_names_resolve():
    for name in optable.__all__:
        assert getattr(optable, name, None) is not None, name


def test_star_import_exports_exactly_all():
    ns = {}
    exec("from optable import *", ns)
    exported = {n for n in ns if n != "__builtins__"}
    assert exported == set(optable.__all__)


def test_no_third_party_names_leak():
    for leaked in ("np", "plt", "scipy", "copy", "csv", "time", "deepcopy"):
        assert leaked not in optable.__all__
