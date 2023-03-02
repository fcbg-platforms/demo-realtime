from functools import partial
from importlib import import_module
from typing import Callable

import pytest


def _requires_module(function: Callable, name: str):
    """Skip a test if package is not available (decorator)."""
    try:
        import_module(name)
        skip = False
    except ImportError:
        skip = True
    reason = f"Test {function.__name__} skipped, requires {name}."
    return pytest.mark.skipif(skip, reason=reason)(function)


requires_psychopy = partial(_requires_module, name="psychopy")
requires_ursina = partial(_requires_module, name="ursina")


def _requires_missing_module(function: Callable, name: str):
    """Skip a test if package is not available (decorator)."""
    try:
        import_module(name)
        skip = True
    except ImportError:
        skip = False
    reason = (
        f"Test {function.__name__} skipped, requires {name} to be missing."
    )
    return pytest.mark.skipif(skip, reason=reason)(function)


requires_missing_psychopy = partial(_requires_missing_module, name="psychopy")
requires_missing_ursina = partial(_requires_missing_module, name="ursina")
