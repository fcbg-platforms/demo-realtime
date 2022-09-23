from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def fft_power(
    data: NDArray[float],
    fs: float,
    band: Tuple[float, float],
    dB: bool,
) -> NDArray[float]:
    """Compute the power of each frequency component represented by the FFT.

    Parameters
    ----------
    data : array of shape (n_channels, n_times)
        Data on which the the FFT is computed.
    fs : float
        Sampling frequency in Hz.
    band : tuple
        Frequency band of interest in Hz as 2 floats, e.g. (8, 13) (edge inc.).
    dB : bool
        If True, the fftval are converted to dB with 10 * np.log10(fftval).

    Returns
    -------
    metric : array of shape (n_channels,)
        Average of power across the frequency component of interest.
    """
    assert data.ndim == 2
    assert len(band) == 2
    assert band[0] <= band[1]
    winsize = data.shape[-1]
    # multiply the data with a window
    window = np.hamming(winsize)
    data = data * window  # 'data *= window' raises if the dtypes are different
    # retrieve fft
    frequencies = np.fft.rfftfreq(winsize, 1 / fs)
    band_idx = np.where((band[0] <= frequencies) & (frequencies <= band[1]))[0]
    fftval = np.abs(np.fft.rfft(data, axis=1)[:, band_idx]) ** 2
    # average across band of interest
    metric = np.average(fftval, axis=1)
    metric = 10 * np.log10(fftval) if dB else metric
    return metric
