from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pytest import Config


def pytest_configure(config: Config) -> None:
    """Configure pytest options."""
    warnings_lines = r"""
    error::
    # numpy deprecations in psychopy c.f. https://github.com/psychopy/psychopy/pull/6247
    ignore:.*Ensure you extract a single element from your array.*:DeprecationWarning
    ignore:`alltrue` is deprecated.*:DeprecationWarning
    """
    for warning_line in warnings_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)


@pytest.fixture(scope="session")
def rng():
    """Random number generator."""
    return np.random.default_rng(seed=101)
