from multiprocessing import Value

from _typeshed import Incomplete
from ursina import Entity, Texture

LANES: list[float]
START_LANE: int
LEFT_EDGE: float
RIGHT_EDGE: float

class Road(Entity):
    """Entity defining a piece of road."""

    def update(self) -> None:
        """Update the position of the piece of road.

        2 pieces are used to simulate the movement sof the car along the road. The 2
        pieces of road are constantly moving down until they exit the screen, at which
        point the road exiting the screen is brought back up.
        """

class Player(Entity):
    """Entity defining the player car.

    Parameters
    ----------
    direction : Value
        Shared variable used to define if the car is going straight (0), left (-1) or
        right (1).
    """

    direction: Incomplete
    hit_edge: bool
    pos_idx: Incomplete

    def __init__(self, direction: Value, *args, **kwargs) -> None: ...
    def update(self) -> None:
        """Update the position of the car and check for collision."""
    x: Incomplete

    def go_right(self) -> None:
        """Move the car one lane to the right."""

    def go_left(self) -> None:
        """Move the car one lane to the left."""

    def go_far_right(self) -> None:
        """Move the car outside the road and back to the far right lane."""

    def go_far_left(self) -> None:
        """Move the car outside the road and back to the far left lane."""

    def safe_shake(self, *args, **kwargs) -> None:
        """Shake the player car if it's not already shaking."""

class Enemy(Entity):
    """Entity defining the ennemy cars."""

    def update(self) -> None:
        """Update the position of the ennemies.

        The traffic has 2 ways, ennemies on the positive x are going up while ennemies
        on the negative x are going down.
        """

def add_enemies(texture: Texture) -> None:
    """Add ennemies on the road.

    A new ennemy is spawn every seconds.

    Parameters
    ----------
    texture : Texture
        Loaded texture of the ennemies car.
    """

def game(direction: Value, enable_ennemies: bool) -> None:
    """Launch the game.

    Parameters
    ----------
    direction : Value
        Shared variable used to define if the car is going straight (0), left (-1) or
        right (1).
    enable_enemies : bool
        If True, enemy cars will spawn.
    """
