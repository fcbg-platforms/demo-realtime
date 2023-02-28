import time
from typing import Optional, Tuple, Union

import numpy as np
from bsl import StreamReceiver
from mne import create_info

from .feedbacks import TopomapMPL
from .metrics import bandpower
from .utils._checks import _check_type
from .utils._docs import fill_doc
from .utils._logs import set_log_level


@fill_doc
def rt_topomap(
    stream_name: str,
    winsize: float = 3,
    duration: float = 30,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    verbose: Optional[Union[str, int]] = None,
):
    """Real-time topographic feedback loop.

    The feedback represents the alpha-band relative power measured by a DSI-24
    amplifier.

    Parameters
    ----------
    %(stream_name)s
    %(winsize)s
    %(duration)s
    %(figsize)s
    %(verbose)s
    """
    set_log_level(verbose)
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

    # retrieve sampling rate and channels
    fs = sr.streams[stream_name].sample_rate
    ch_names = sr.streams[stream_name].ch_list
    # remove unwanted channels
    ch2remove = ("TRIGGER", "TRG", "X1", "X2", "X3", "A1", "A2")
    ch_idx = np.array(
        [k for k, ch in enumerate(ch_names) if ch not in ch2remove]
    )
    # filter channel name list
    ch_names = [ch for ch in ch_names if ch not in ch2remove]

    # create feedback
    info = create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    info.set_montage("standard_1020")
    feedback = TopomapMPL(info, "Purples", figsize)

    # wait to fill one buffer
    time.sleep(winsize)

    # main loop
    start = time.time()
    while time.time() - start <= duration:
        # retrieve data
        sr.acquire()
        data, _ = sr.get_window()
        # compute metric
        metric = bandpower(
            data[:, ch_idx].T, fs=fs, method="periodogram", band=(8, 13)
        )
        # update feedback
        feedback.update(metric)

    # close the feedback window
    feedback.close()
