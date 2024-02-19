import re
from pathlib import Path

import pytest

from demo_realtime import logger, set_log_level
from demo_realtime.visuals import DoubleSpinningWheel

set_log_level("INFO")
logger.propagate = True


def test_double_spinning_wheel():
    """Test the double spinning wheel feedback."""
    pytest.importorskip("psychopy")
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


def test_invalid_double_spinning_wheel():
    """Test the double spinning wheel feedback with invalid arguments."""
    pytest.importorskip("psychopy")
    with pytest.raises(ValueError, match="should be 'norm'"):
        DoubleSpinningWheel(units="101")

    with pytest.warns(RuntimeWarning, match="'pyglet' window type is recommended"):
        viz = DoubleSpinningWheel(winType="101")
    del viz
    with pytest.warns(RuntimeWarning, match=re.escape("'(-1, -1, -1)' is recommended")):
        viz = DoubleSpinningWheel(color="101")
    del viz

    with pytest.raises(TypeError, match="must be an instance of"):
        DoubleSpinningWheel(wheel_size="101")
    with pytest.raises(TypeError, match="must be an instance of"):
        DoubleSpinningWheel(offset="101")

    with pytest.warns(RuntimeWarning, match="'wheel_size' should be in the range"):
        viz = DoubleSpinningWheel(wheel_size=101)
    del viz
    with pytest.warns(RuntimeWarning, match="'offset' should be in the range"):
        viz = DoubleSpinningWheel(offset=101)
    del viz
