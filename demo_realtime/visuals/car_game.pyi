from _typeshed import Incomplete

from ..utils._checks import check_type as check_type
from ..utils._imports import import_optional_dependency as import_optional_dependency

_DIRECTION_MAPPING: Incomplete
_DIRECTION_MAPPING_INV: Incomplete

class CarGame:
    """A simple 4-lane car game where the player has to dodge traffic.

    The player can move the car to the left or right with :meth:`CarGame.go_direction`.

    Parameters
    ----------
    enable_enemies : bool
        If True, enemy cars will spawn.
    """

    _enable_enemies: Incomplete

    def __init__(self, enable_enemies: bool = True) -> None: ...
    def start(self) -> None:
        """Start the game."""

    def stop(self) -> None:
        """Stop the game and close the game window."""
    _direction: Incomplete
    _process: Incomplete

    def _create_shared_variables(self) -> None:
        """Create the process and shared variables."""

    def go_direction(self, direction: str) -> None:
        """Move the player car in a given direction."""

    @property
    def direction(self) -> str:
        """Direction in which the player car is going."""
