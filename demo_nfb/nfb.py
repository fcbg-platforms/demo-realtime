import time

import numpy as np
from bsl import StreamReceiver
from stimuli.visuals import FillingBar

from . import fft
from .utils._checks import _check_type


def nfb_fft_alpha_occipital(
    stream_name: str, winsize: float = 3, duration: float = 30
) -> None:
    """Run a simple neurofeedback loop.

    The feedback represents the alpha-band FFT on the O1 and O2 electrodes.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    winsize : float
        Duration of an acquisition window.
    duration : float
        Duration of the neurofeedback loop.
    """
    # check inputs
    _check_type(stream_name, (str,), "stream_name")
    _check_type(winsize, ("numeric",), "winsize")
    assert 0 < winsize
    _check_type(duration, ("numeric",), "duration")
    assert 0 < duration

    # create receiver and feedback
    sr = StreamReceiver(
        bufsize=winsize, winsize=winsize, stream_name=stream_name
    )
    feedback = FillingBar(window_size=(1280, 720))
    feedback.draw_background("lightgrey")
    feedback.putBar(400, 50, 5, "black", "teal", axis=1)  # empty bar

    # store 100 points to compute percentile for min/max
    metrics = np.ones(100) * np.nan
    inc = 0

    # retrieve sampling rate and channels
    fs = sr.streams[stream_name].sample_rate
    ch_names = sr.streams[stream_name].ch_list
    ch_idx = [k for k, ch in enumerate(ch_names) if ch in ("O1", "O2")]

    # wait to fill one buffer
    time.sleep(winsize)

    # main loop
    start = time.time()
    while time.time() - start <= duration:
        # retrieve data
        sr.acquire()
        data, _ = sr.get_window()
        # compute metric
        metric = fft(data[:, ch_idx].T, fs=fs, band=(8, 13))

        # store metric
        metrics[inc % 100] = metric
        inc += 1
        # skip until we have enough points to define the feedback range
        if inc < metrics.size:
            start = time.time()  # reset start time
            continue
        # define the feedback range
        min_ = np.percentile(metrics, 5)
        max_ = np.percentile(metrics, 95)
        fill_perc = (metric - min_) / (max_ - min_)
        fill_perc = np.clip(fill_perc, 0, 1)

        # update feedback
        feedback.fill_perc = fill_perc
        feedback.show()

    # close the feedback window
    feedback.close()
