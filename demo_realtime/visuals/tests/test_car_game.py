import time

import pytest

from demo_realtime.visuals import CarGame


def test_car_game():
    """Test the car game feedback."""
    pytest.importorskip("ursina")
    game = CarGame(enable_enemies=True)
    game.start()
    time.sleep(4)
    assert game._process.is_alive()
    game.go_direction("right")
    assert game.direction == "right"
    with pytest.warns(RuntimeWarning, match="Already going right."):
        game.go_direction("right")
    assert game.direction == "right"
    time.sleep(1)
    game.go_direction("right")
    assert game.direction == "right"
    with pytest.warns(RuntimeWarning, match="Already going right."):
        game.go_direction("left")
    time.sleep(1)
    game.stop()
    time.sleep(1)
    with pytest.warns(RuntimeWarning, match="is not running"):
        game.go_direction("right")
    with pytest.warns(RuntimeWarning, match="is not running"):
        game.go_direction("left")
    game.start()
    time.sleep(4)
    assert game._process.is_alive()
    game.go_direction("left")
    with pytest.warns(RuntimeWarning, match="Already going left."):
        game.go_direction("left")
    game.stop()

    game = CarGame(enable_enemies=False)
    game.start()
    time.sleep(2)
    assert game._process.is_alive()
    assert game.direction == "straight"
    game.go_direction("right")
    assert game.direction == "right"
    time.sleep(1)
    assert game.direction == "straight"
    game.go_direction("left")
    assert game.direction == "left"
    game.stop()
