from pathlib import Path

import pytest

from demo_realtime import logger, set_log_level
from demo_realtime.utils._tests import requires_module
from demo_realtime.visuals import DoubleSpinningWheel

set_log_level("INFO")
logger.propagate = True


@requires_module("psychopy")
def test_double_spinning_wheel():
    """Test the double spinning wheel feedback."""
    viz = DoubleSpinningWheel()
    assert isinstance(viz.image, Path)
    assert viz._status.value == 0
    assert not viz._process.is_alive()
    viz.start()
    assert viz._status.value == 1
    assert viz._process.is_alive()
    assert viz.speed == 0
    viz.speed = 2
    assert viz.speed == 2
    assert viz._status.value == 1
    assert viz._process.is_alive()
    assert viz.active is True
    viz.stop()
    assert viz._status.value == 0
    assert not viz._process.is_alive()
    assert viz.active is False

    with pytest.raises(RuntimeError, match="already stopped"):
        viz.stop()

    viz = DoubleSpinningWheel()
    viz.start()
    with pytest.raises(RuntimeError, match="already started"):
        viz.start()
    del viz


@requires_module("psychopy")
def test_invalid_double_spinning_wheel(caplog):
    """Test the double spinning wheel feedback with invalid arguments."""
    with pytest.raises(ValueError, match="should be 'norm'"):
        DoubleSpinningWheel(units="101")

    caplog.clear()
    viz = DoubleSpinningWheel(winType="101")
    assert "'pyglet' window type is recommended" in caplog.text
    del viz
    caplog.clear()
    viz = DoubleSpinningWheel(color=(0, 0, 0))
    assert "is recommended" in caplog.text
    caplog.clear()
    del viz

    with pytest.raises(TypeError, match="must be an instance of"):
        DoubleSpinningWheel(wheel_size="101")
    with pytest.raises(TypeError, match="must be an instance of"):
        DoubleSpinningWheel(offset="101")

    caplog.clear()
    viz = DoubleSpinningWheel(wheel_size=101)
    assert "size should be" in caplog.text
    caplog.clear()
    del viz
    viz = DoubleSpinningWheel(offset=101)
    assert "offset should be" in caplog.text
    caplog.clear()
    del viz
