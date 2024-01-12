from multiprocessing import Process, Value
from warnings import warn

from ..utils._checks import check_type
from ..utils._imports import import_optional_dependency
from ..utils.logs import logger

_DIRECTION_MAPPING = {
    -1: "left",
    0: "straight",
    1: "right",
}
_DIRECTION_MAPPING_INV = {value: key for key, value in _DIRECTION_MAPPING.items()}


class CarGame:
    """A simple 4-lane car game where the player has to dodge traffic.

    The player can move the car to the left or right with :meth:`CarGame.go_direction`.

    Parameters
    ----------
    enable_enemies : bool
        If True, enemy cars will spawn.
    """

    def __init__(self, enable_enemies: bool = True) -> None:
        import_optional_dependency("ursina")
        check_type(enable_enemies, (bool,), "enable_enemies")
        self._enable_enemies = enable_enemies
        # prepare shared variables and process to control the game
        self._create_shared_variables()

    def start(self) -> None:
        """Start the game."""
        if self._process.is_alive():
            raise RuntimeError("The game is already started.")
        self._process.start()

    def stop(self) -> None:
        """Stop the game and close the game window."""
        if not self._process.is_alive():
            raise RuntimeError("The game is already stopped.")
        self._process.kill()  # not clean, but works like a charm
        # prepare a restart
        self._create_shared_variables()

    def _create_shared_variables(self):
        """Create the process and shared variables."""
        from ._car_game import game

        self._direction = Value("i", 0)  # -1: left, 0: straight, 1: right
        self._process = Process(
            target=game,
            args=(self._direction, self._enable_enemies),
        )

    def go_direction(self, direction: str) -> None:
        """Move the player car in a given direction."""
        try:
            direction_int = _DIRECTION_MAPPING_INV[direction]
        except KeyError:
            raise RuntimeError(f"The provided 'direction' {direction} is not valid.")
        if self._process.is_alive():
            if self._direction.value == 0:
                with self._direction.get_lock():
                    self._direction.value = direction_int
            else:
                warn(
                    f"Already going {self.direction}. Command '{direction}' ignored.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warn(
                f"The game is not running. Command '{direction}' ignored.",
                RuntimeWarning,
                stacklevel=2,
            )

    # -------------------------------------------------------------------------
    @property
    def direction(self) -> str:
        """Direction in which the player car is going."""
        return _DIRECTION_MAPPING[self._direction.value]
