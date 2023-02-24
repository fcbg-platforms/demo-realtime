import re

import numpy as np
import pytest

from demo_realtime.metrics import bandpower

# generate fake signal with known frequency content @ 8 Hz and 20 Hz
rng = np.random.default_rng(seed=101)
fs = 1000
n_channels = 5
times = np.arange(0, 4, 1 / fs)
sources = np.vstack(
    (np.sin(8 * 2 * np.pi * times), np.sin(20 * 2 * np.pi * times))
)
weights = rng.uniform(low=0.5, high=4, size=(n_channels, 2))
data = weights @ sources
data += rng.normal(loc=np.mean(data), scale=np.std(data) / 3, size=data.shape)


@pytest.mark.parametrize("method", ("periodogram", "welch", "multitaper"))
def test_relative_bandpower(method):
    """Test the relative bandpower."""
    kwargs = dict(nperseg=fs) if method == "welch" else dict()
    bp = bandpower(data, fs, method, band=(0, 500), relative=True, **kwargs)
    assert np.allclose(bp, np.ones(n_channels))

    bp1 = bandpower(data, fs, method, band=(7, 9), relative=True, **kwargs)
    bp2 = bandpower(data, fs, method, band=(19, 21), relative=True, **kwargs)
    assert np.allclose(bp1 + bp2, np.ones(n_channels), atol=0.3)


@pytest.mark.parametrize("method", ("periodogram", "welch", "multitaper"))
def test_absolute_bandpower(method):
    """Test dB set to True."""
    kwargs = dict(nperseg=fs) if method == "welch" else dict()
    bp1 = bandpower(data, fs, method, band=(7, 9), relative=True, **kwargs)
    bp2 = bandpower(data, fs, method, band=(19, 21), relative=True, **kwargs)
    ratio1 = bp1 / bp2
    bp1 = bandpower(data, fs, method, band=(7, 9), relative=False, **kwargs)
    bp2 = bandpower(data, fs, method, band=(19, 21), relative=False, **kwargs)
    ratio2 = bp1 / bp2
    assert np.allclose(ratio1, ratio2)


def test_invalid():
    """Test invalid inputs."""
    with pytest.raises(AssertionError, match="must be a 2D array"):
        bandpower(rng.random((3, 1000, 2)), 100, "periodogram", band=(0, 500))
    with pytest.raises(RuntimeError, match="'101' is not supported."):
        bandpower(data, fs, method="101", band=(8, 13))
    with pytest.raises(AssertionError, match="must be a 2-length tuple."):
        bandpower(data, fs, "periodogram", band=(8, 13, 17))
    with pytest.raises(
        AssertionError, match=re.escape("must be defined as (low, high)")
    ):
        bandpower(data, fs, "periodogram", band=(13, 8))
