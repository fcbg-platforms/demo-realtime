from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest import Config


def pytest_configure(config: Config) -> None:
    """Configure pytest options."""
    warnings_lines = r"""
    error::
    """
    for warning_line in warnings_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
