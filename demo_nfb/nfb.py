import time

from bsl import StreamReceiver
from stimuli.visuals import FillingBar

from . import fft


def basic(stream_name: str):
    """A basic NFB loop that runs 30 seconds.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    """
    # create receiver and feedback
    sr = StreamReceiver(bufsize=1, winsize=1, stream_name=stream_name)
    feedback = FillingBar()
    feedback.draw_background("lightgrey")
    feedback.putBar(400, 50, 5, "black", "teal", axis=1)  # empty bar

    # init min/max for percentage
    min_ = None
    max_ = None

    # retrieve sampling rate
    fs = sr.streams[stream_name].sample_rate

    # wait to fill one buffer
    time.sleep(1)

    # loop for 30 seconds
    start = time.time()
    while time.time() - start <= 30:
        # retrieve data
        sr.acquire()
        data, _ = sr.get_window()
        # compute metric
        metric = fft(data.T, fs=fs, band=(8, 13))

        # update feedback from metric
        if min_ is None and max_ is None:
            fill_perc = 0.5
            min_ = metric
            max_ = metric
        else:
            if metric < min_:
                min_ = metric
            if metric > max_:
                max_ = metric
            fill_perc = (metric - min_) / (max_ - min_)
        feedback.fill_perc = fill_perc
        feedback.show()

    # close the feedback window
    feedback.close()
