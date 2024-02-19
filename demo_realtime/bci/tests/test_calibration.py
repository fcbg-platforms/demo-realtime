from pathlib import Path

import pytest
from mne import find_events
from mne.io import read_raw_fif
from mne_lsl.player import PlayerLSL as Player

from demo_realtime.bci import calibration


@pytest.fixture(scope="module")
def mock_dsi_stream(request):
    """Create a mock DSI stream."""
    fname = Path(__file__).parents[3] / "data" / "dsi-24-raw.fif"
    if not fname.exists():
        pytest.skip("DSI-24 recording not found.")
    with Player(fname, name=f"P_{request.node.name}", annotations=False) as player:
        yield player


@pytest.mark.xfail(run=False, reason="Unraisable exception during teardown.")
def test_calibration(tmp_path, mock_dsi_stream):
    """Test the calibration."""
    pytest.importorskip("psychopy")
    fname = calibration(2, mock_dsi_stream.name, tmp_path, skip_instructions=True)
    raw = read_raw_fif(fname, preload=True)
    assert "TRG" not in raw.ch_names
    assert all(ch in raw.ch_names for ch in ["X1", "X2", "X3", "A2"])
    raw.drop_channels(["X1", "X2", "X3", "A2"])
    assert "TRIGGER" in raw.ch_names
    events = find_events(raw, stim_channel="TRIGGER")
    assert events.shape == (6, 3)
    assert sorted(events[:, 2]) == [1, 1, 2, 2, 3, 3]
    raw.drop_channels("TRIGGER")
    assert len(raw.ch_names) == 19  # number of EEG channels
    raw.set_montage("standard_1020")
