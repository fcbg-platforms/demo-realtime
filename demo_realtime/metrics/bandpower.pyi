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
