from multiprocessing import Process, Value

from ..utils._imports import import_optional_dependency
from ..utils._logs import logger


class CarGame:
    """A simple 4-lane car game where the player has to dodge traffic.

    The player can move the car to the left or right with `CarGame.go_left` and
    `CarGame.go_right`.
    """

    def __init__(self) -> None:
        import_optional_dependency("ursina")
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
            args=(self._direction,),
        )

    def go_left(self) -> None:
        """Move the player car one lane to the left."""
        if self._process.is_alive():
            if self._direction.value == 0:
                logger.debug("Setting direction to -1.")
                with self._direction.get_lock():
                    self._direction.value = -1
            else:
                logger.warning(
                    "Already going %s. Command ignored.", self.direction
                )
        else:
            logger.warning("The game is not running. Command ignored.")

    def go_right(self) -> None:
        """Move the player car one lane to the right."""
        if self._process.is_alive():
            if self._direction.value == 0:
                logger.debug("Setting direction to 1.")
                with self._direction.get_lock():
                    self._direction.value = 1
            else:
                logger.warning(
                    "Already going %s. Command ignored.", self.direction
                )
        else:
            logger.warning("The game is not running. Command ignored.")

    # -------------------------------------------------------------------------
    @property
    def direction(self) -> str:
        """Direction in which the player car is going."""
        mapping = {
            -1: "left",
            0: "straight",
            1: "right",
        }
        return mapping[self._direction.value]
