from platform import system

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mne import create_info
from mne.channels import make_standard_montage

from demo_realtime import logger, set_log_level
from demo_realtime.visuals import TopomapMPL

set_log_level("INFO")
logger.propagate = True


@pytest.mark.skipif(
    system() == "Linux",
    reason="Interactive QtAgg backend not supported in Linux CIs.",
)
def test_topomap():
    """Test the topographic map feedback."""
    montage = make_standard_montage("biosemi32")
    info = create_info(montage.ch_names, sfreq=1, ch_types="eeg")
    info.set_montage("biosemi32")
    rng = np.random.default_rng(seed=101)

    viz = TopomapMPL(info)
    viz.update(rng.random(size=len(info.ch_names)))
    viz.update(rng.random(size=len(info.ch_names)))
    viz.update(rng.random(size=len(info.ch_names)))

    assert isinstance(viz.fig, plt.Figure)
    assert isinstance(viz.axes, Axes)
    assert isinstance(viz.cmap, str)
    viz.close()


@pytest.mark.skipif(
    system() == "Linux",
    reason="Interactive QtAgg backend not supported in Linux CIs.",
)
def test_invalid_topomap(caplog):
    """Test the topographic map feedback with invalid arguments."""
    montage = make_standard_montage("biosemi32")
    info = create_info(montage.ch_names, sfreq=1, ch_types="eeg")
    with pytest.raises(ValueError, match="does not have a DigMontage"):
        TopomapMPL(info)
    info.set_montage("biosemi32")
    with pytest.raises(TypeError, match="must be an instance of"):
        TopomapMPL(info, cmap=101)
    with pytest.raises(ValueError, match="should be a 2-item tuple"):
        TopomapMPL(info, figsize=(101, 101, 101))
    with pytest.raises(TypeError, match="must be an instance of"):
        TopomapMPL(info, figsize="10")
    with pytest.raises(ValueError, match="strictly positive"):
        TopomapMPL(info, figsize=(-101, 101))

    caplog.clear()
    viz = TopomapMPL(info, figsize=(2, 3))
    assert "square" in caplog.text
    del viz
    caplog.clear()
    viz = TopomapMPL(info, figsize=(101, 101))
    assert "Large figsize" in caplog.text
    del viz
    caplog.clear()
