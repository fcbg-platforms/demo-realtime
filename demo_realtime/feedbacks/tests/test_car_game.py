import time

import pytest

from demo_realtime import logger, set_log_level
from demo_realtime.feedbacks import CarGame
from demo_realtime.utils._tests import requires_missing_ursina, requires_ursina

set_log_level("WARNING")
logger.propagate = True


@requires_missing_ursina
def test_missing_psychopy():
    """Test error if Ursina is missing."""
    with pytest.raises(ImportError, match="optional dependency 'ursina'"):
        CarGame()


@requires_ursina
def test_car_game(caplog):
    """Test the car game feedback."""
    game = CarGame()
    game.start()

    time.sleep(2)
    game.go_right()
    assert game.direction == "right"
    caplog.clear()

    game.go_right()
    assert game.direction == "right"
    assert "Already going right." in caplog.text
    caplog.clear()

    time.sleep(2)
    game.go_right()
    assert game.direction == "right"
    assert "Already going right." not in caplog.text
    caplog.clear()

    game.go_left()
    assert "Already going right." in caplog.text
    caplog.clear()

    game.stop()

    time.sleep(2)
    game.go_left()
    assert "The game is not running." in caplog.text
    caplog.clear()

    game.go_right()
    assert "The game is not running." in caplog.text
    caplog.clear()

    game.start()
    time.sleep(2)
    assert game.direction == "straight"
    game.go_right()
    assert game.direction == "right"
    assert "Already going right." not in caplog.text
    game.stop()
