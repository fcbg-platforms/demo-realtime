"""Test _imports.py"""

import pytest

from .._imports import _import_optional_dependency


def test_import_optional_dependency():
    """Test the import of optional dependencies."""
    # Test import of present package
    numpy = _import_optional_dependency("numpy")
    assert isinstance(numpy.__version__, str)

    # Test import of absent package
    with pytest.raises(ImportError, match="Missing optional dependency"):
        _import_optional_dependency("non_existing_pkg", raise_error=True)

    # Test import of absent package without raise
    pkg = _import_optional_dependency("non_existing_pkg", raise_error=False)
    assert pkg is None

    # Test extra
    with pytest.raises(ImportError, match="blabla"):
        _import_optional_dependency("non_existing_pkg", extra="blabla")
