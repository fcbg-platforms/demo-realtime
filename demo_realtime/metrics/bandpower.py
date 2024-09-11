from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simpson
from scipy.signal import periodogram, welch

if TYPE_CHECKING:
    from numpy.typing import NDArray


def bandpower(
    data: NDArray[float],
    fs: float,
    method: str,
    band: tuple[float, float],
    relative: bool = True,
    **kwargs,
) -> NDArray[float]:
    """Compute the bandpower of the individual channels.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
        Data on which the the bandpower is estimated.
    fs : float
        Sampling frequency in Hz.
    method : ``'periodogram'`` | ``'welch'`` | ``multitaper``
        Method used to estimate the power spectral density.
    band : tuple of shape (2,)
        Frequency band of interest in Hz as 2 floats, e.g. ``(8, 13)``. The
        edges are included.
    relative : bool
        If True, the relative bandpower is returned instead of the absolute
        bandpower.
    **kwargs : dict
        Additional keyword arguments are provided to the power spectral density
        estimation function.
        * ``'periodogram'``: `scipy.signal.periodogram`
        * ``'welch'``: `scipy.signal.welch`
        * ``multitaper``: `mne.time_frequency.psd_array_multitaper`

        The only provided arguments are the data array and the sampling
        frequency.

    Returns
    -------
    bandpower : array of shape (n_channels,)
        The bandpower of each channel.
    """
    # compute the power spectral density
    assert (
        data.ndim == 2
    ), "The provided data must be a 2D array of shape (n_channels, n_samples)."

    if method == "periodogram":
        freqs, psd = periodogram(data, fs, **kwargs)
    elif method == "welch":
        freqs, psd = welch(data, fs, **kwargs)
    elif method == "multitaper":
        psd, freqs = psd_array_multitaper(data, fs, **kwargs)
    else:
        raise RuntimeError(f"The provided method '{method}' is not supported.")

    # compute the bandpower
    assert len(band) == 2, "The 'band' argument must be a 2-length tuple."
    assert (
        band[0] <= band[1]
    ), "The 'band' argument must be defined as (low, high) (in Hz)."
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bandpower = simpson(psd[:, idx_band], dx=freq_res)
    bandpower = bandpower / simpson(psd, dx=freq_res) if relative else bandpower
    return bandpower
