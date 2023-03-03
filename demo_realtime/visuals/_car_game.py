from multiprocessing import Value
from typing import List

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

import numpy as np
from ursina import (
    Entity,
    Texture,
    Ursina,
    camera,
    color,
    destroy,
    invoke,
    raycast,
    time,
)

LANES: List[float] = [-3.51, -1.17, 1.17, 3.51]
START_LANE: int = 1
LEFT_EDGE: float = -4.6
RIGHT_EDGE: float = 4.6


class Road(Entity):
    """Entity defining a piece of road."""

    def update(self) -> None:
        """Update the position of the piece of road.

        2 pieces are used to simulate the movement sof the car along the road.
        The 2 pieces of road are constantly moving down until they exit the
        screen, at which point the road exiting the screen is brought back up.
        """
        self.y -= 6 * time.dt
        if self.y < -15:
            self.y += 30


class Player(Entity):
    """Entity defining the player car.

    Parameters
    ----------
    direction : Value
        Shared variable used to define if the car is going straight (0),
        left (-1) or right (1).
    """

    def __init__(self, direction: Value, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # define variables used to control the car position
        assert direction.value == 0  # sanity-check
        self.direction = direction
        self.hit_edge = False
        self.pos_idx = START_LANE

    def update(self) -> None:
        """Update the position of the car and check for collision."""
        # handle the changes in direction
        if self.direction.value == 1 and self.pos_idx == len(LANES) - 1:
            # go right a bit, shake, and go back to the far right lane
            self.go_far_right()
        elif self.direction.value == 1 and self.x < LANES[-1]:
            self.go_right()
        elif self.direction.value == -1 and self.pos_idx == 0:
            # go left a bit, shake, and go back to the far left lane
            self.go_far_left()
        elif self.direction.value == -1 and LANES[0] < self.x:
            self.go_left()

        # handle collision with ennemies
        hit_infos = [
            raycast(self.world_position, (0.5, 1, 0), distance=1),
            raycast(self.world_position, (-0.5, 1, 0), distance=1),
            raycast(self.world_position, (0.5, -1, 0), distance=1),
            raycast(self.world_position, (-0.5, -1, 0), distance=1),
        ]
        if any(hit_info.hit for hit_info in hit_infos):
            self.safe_shake(duration=0.3, magnitude=3)

    def go_right(self) -> None:
        """Move the car one lane to the right."""
        self.x += 5 * time.dt
        if np.isclose(self.x, LANES[self.pos_idx + 1], atol=0.15):
            self.pos_idx += 1
            self.x = LANES[self.pos_idx]
            # reset variables
            with self.direction.get_lock():
                self.direction.value = 0

    def go_left(self) -> None:
        """Move the car one lane to the left."""
        self.x -= 5 * time.dt
        if np.isclose(self.x, LANES[self.pos_idx - 1], atol=0.15):
            self.pos_idx -= 1
            self.x = LANES[self.pos_idx]
            # reset variables
            with self.direction.get_lock():
                self.direction.value = 0

    def go_far_right(self) -> None:
        """Move the car outside the road and back to the far right lane."""
        if self.hit_edge:
            self.x -= 5 * time.dt
        else:
            self.x += 5 * time.dt
        if RIGHT_EDGE <= self.x:
            self.hit_edge = True
            self.safe_shake(magnitude=1)

        # break condition
        if self.hit_edge and np.isclose(self.x, LANES[-1], atol=0.15):
            self.x = LANES[-1]
            # reset variables
            self.hit_edge = False
            with self.direction.get_lock():
                self.direction.value = 0

    def go_far_left(self) -> None:
        """Move the car outside the road and back to the far left lane."""
        if self.hit_edge:
            self.x += 5 * time.dt
        else:
            self.x -= 5 * time.dt
        if self.x <= LEFT_EDGE:
            self.hit_edge = True
            self.safe_shake(magnitude=1)

        # break condition
        if self.hit_edge and np.isclose(self.x, LANES[0], atol=0.15):
            self.x = LANES[0]
            # reset variables
            self.hit_edge = False
            with self.direction.get_lock():
                self.direction.value = 0

    def safe_shake(self, *args, **kwargs) -> None:
        """Shake the player car if it's not already shaking."""
        if hasattr(self, "shake_sequence") and self.shake_sequence:
            finished = getattr(getattr(self, "shake_sequence"), "finished")
            if finished:
                self.shake(*args, **kwargs)
        else:
            self.shake(*args, **kwargs)


class Enemy(Entity):
    """Entity defining the ennemy cars."""

    def update(self) -> None:
        """Update the position of the ennemies.

        The traffic has 2 ways, ennemies on the positive x are going up while
        ennemies on the negative x are going down.
        """
        if self.x < 0:
            self.y -= 8 * time.dt
        else:
            self.y -= 5 * time.dt
        if self.y < -10:
            destroy(self)


def add_enemies(texture: Texture) -> None:
    """Add ennemies on the road.

    A new ennemy is spawn every seconds.

    Parameters
    ----------
    texture : Texture
        Loaded texture of the ennemies car.
    """
    x = np.random.choice(LANES)
    Enemy(
        model="quad",
        texture=texture,
        collider="box",
        scale=(2, 1),
        x=x,
        y=25,
        color=color.random_color(),
        rotation_z=90 if x < 0 else -90,
    )
    invoke(add_enemies, texture, delay=1)  # call itself to spawn infinitely


def game(direction: Value, enable_ennemies: bool) -> None:
    """Launch the game.

    Parameters
    ----------
    direction : Value
        Shared variable used to define if the car is going straight (0),
        left (-1) or right (1).
    enable_enemies : bool
        If True, enemy cars will spawn.
    """
    # create application
    app = Ursina()
    # set camera angle and FOV
    camera.orthographic = True
    camera.fov = 10

    # create entities
    texture = Texture(
        files("demo_realtime.visuals") / "resources" / "road.png"
    )
    Road(model="quad", texture=texture, scale=15, z=1)
    Road(model="quad", texture=texture, scale=15, z=1, y=15)
    texture = Texture(
        files("demo_realtime.visuals") / "resources" / "car-player.png"
    )
    Player(
        direction=direction,
        model="quad",
        texture=texture,
        scale=(2, 1),
        rotation_z=-90,
        x=LANES[START_LANE],  # 1.17, 3.51
        y=-3,
    )
    # load texture for ennemy cars
    car_enemy_texture = Texture(
        files("demo_realtime.visuals") / "resources" / "car-enemy.png"
    )

    # start the app
    if enable_ennemies:
        add_enemies(car_enemy_texture)
    app.run()
