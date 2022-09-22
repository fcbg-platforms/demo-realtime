import time

import numpy as np
from bsl import StreamReceiver
from stimuli.visuals import FillingBar

from . import fft


def basic(stream_name: str) -> None:
    """Run a 30 second neurofeedback loop.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    """
    # create receiver and feedback
    sr = StreamReceiver(bufsize=3, winsize=3, stream_name=stream_name)
    feedback = FillingBar()
    feedback.draw_background("lightgrey")
    feedback.putBar(400, 50, 5, "black", "teal", axis=1)  # empty bar

    # store 100 points to compute percentile for min/max
    metrics = np.ones(100) * np.nan
    inc = 0

    # retrieve sampling rate
    fs = sr.streams[stream_name].sample_rate

    # wait to fill one buffer
    time.sleep(3)

    # loop for 30 seconds
    start = time.time()
    while time.time() - start <= 30:
        # retrieve data
        sr.acquire()
        data, _ = sr.get_window()
        # compute metric
        metric = fft(data.T, fs=fs, band=(8, 13))

        # store metric and update feedback range
        metrics[inc % 100] = metric
        inc += 1
        if inc < metrics.size:
            continue  # skip until we have 100 points
        min_ = np.percentile(metrics, 5)
        max_ = np.percentile(metrics, 95)
        fill_perc = (metric - min_) / (max_ - min_)
        fill_perc = np.clip(fill_perc, 0, 1)

        # update feedback
        feedback.fill_perc = fill_perc
        feedback.show()

    # close the feedback window
    feedback.close()
