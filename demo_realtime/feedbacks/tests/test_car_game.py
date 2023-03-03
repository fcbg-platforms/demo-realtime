import time

import pytest

from demo_realtime import logger, set_log_level
from demo_realtime.feedbacks import CarGame
from demo_realtime.utils._tests import requires_missing_ursina, requires_ursina

set_log_level("WARNING")
logger.propagate = True


@requires_missing_ursina
def test_missing_ursina():
    """Test error if Ursina is missing."""
    with pytest.raises(ImportError, match="optional dependency 'ursina'"):
        CarGame()


@requires_ursina
def test_car_game(caplog):
    """Test the car game feedback."""
    game = CarGame(enable_enemies=True)
    game.start()
    time.sleep(4)
    assert game._process.is_alive()
    game.go_right()
    assert game.direction == "right"
    caplog.clear()
    game.go_right()
    assert game.direction == "right"
    assert "Already going right." in caplog.text
    caplog.clear()
    time.sleep(1)
    game.go_right()
    assert game.direction == "right"
    game.go_left()
    assert "Already going right." in caplog.text
    time.sleep(1)
    game.stop()
    time.sleep(1)
    caplog.clear()
    game.go_right()
    assert "is not running" in caplog.text
    caplog.clear()
    game.go_left()
    assert "is not running" in caplog.text
    caplog.clear()
    game.start()
    time.sleep(4)
    assert game._process.is_alive()
    game.go_left()
    assert "Already going left." not in caplog.text
    game.go_left()
    assert "Already going left." in caplog.text
    game.stop()

    game = CarGame(enable_enemies=False)
    game.start()
    time.sleep(4)
    assert game._process.is_alive()
    assert game.direction == "straight"
    game.go_right()
    assert game.direction == "right"
    time.sleep(1)
    assert game.direction == "straight"
    game.go_left()
    assert game.direction == "left"
    game.stop()
