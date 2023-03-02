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
    assert game._process.is_alive()
    # game.go_right()
    # assert game.direction == "right"
    # caplog.clear()
    # game.go_right()
    # assert game.direction == "right"
    # assert "Already going right." in caplog.text
    # caplog.clear()
    # time.sleep(1)
    # game.go_right()
    # assert game.direction == "right"
    # game.go_left()
    # assert "Already going right." in caplog.text
