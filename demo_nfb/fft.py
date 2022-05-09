from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def fft(data: NDArray[float], fs: float, band: Tuple[float, float]):
    """Apply FFT to the data after applying a hamming window.

    Parameters
    ----------
    data : array
        2D array of shape (n_channels, n_times) containing the received data.
    fs : float
        Sampling frequency in Hz.
    band : tuple
        Frequency band of interest in Hz as 2 floats, e.g. (8, 13) (edge inc.).

    Returns
    -------
    metric : float
        Average of abs(FFT) across channel and frequency binds in the band of
        interest.
    """
    assert data.ndim == 2
    winsize = data.shape[-1]
    # mutliply the data with a window
    window = np.hamming(winsize)
    data = data * window
    # retrieve fft
    frequencies = np.fft.rfftfreq(winsize, 1 / fs)
    assert len(band) == 2
    assert band[0] <= band[1]
    band_idx = np.where((band[0] <= frequencies) & (frequencies <= band[1]))[0]
    fftval = np.abs(np.fft.rfft(data, axis=-1))
    # average across channels and band of interest
    metric = np.average(fftval[:, band_idx])
    return metric
