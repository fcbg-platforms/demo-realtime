from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest options."""
    warnings_lines = r"""
    error::
    """
    for warning_line in warnings_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)


@pytest.fixture(scope="session")
def rng():
    """Random number generator."""
    return np.random.default_rng(seed=101)
