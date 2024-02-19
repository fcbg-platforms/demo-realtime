import time

import numpy as np
from mne_lsl.stream import StreamLSL

from .metrics import bandpower
from .utils._checks import check_type
from .utils._docs import fill_doc
from .utils._imports import import_optional_dependency
from .utils.logs import verbose


@fill_doc
@verbose
def nfb_filling_bar(
    stream_name: str,
    winsize: float = 3,
    duration: float = 30,
    *,
    verbose: bool | str | int | None = None,
) -> None:
    """Real-time neurofeedback loop using a feedback horizontal filling bar.

    The feedback represents the alpha-band power on the occipital electrodes
    ``O1`` and ``O2``.

    Parameters
    ----------
    %(stream_name)s
    %(winsize)s
    %(duration)s
    %(verbose)s
    """
    import_optional_dependency("stimuli")

    from stimuli.visuals import FillingBar

    # check inputs
    check_type(stream_name, (str,), "stream_name")
    check_type(winsize, ("numeric",), "winsize")
    assert 0 < winsize
    check_type(duration, ("numeric",), "duration")
    assert 0 < duration

    # create receiver and feedback
    stream = StreamLSL(bufsize=winsize, name=stream_name).connect()
    stream.pick(("O1", "O2"))
    feedback = FillingBar(window_size=(1280, 720))
    feedback.draw_background("lightgrey")
    feedback.putBar(400, 50, 5, "black", "teal", axis=1)  # empty bar

    # store 100 points to compute percentile for min/max
    metrics = np.ones(100) * np.nan
    inc = 0

    # wait to fill one buffer
    time.sleep(winsize)

    # main loop
    start = time.time()
    while time.time() - start <= duration:
        data, _ = stream.get_data()
        # compute metric
        metric = bandpower(
            data, fs=stream.info["sfreq"], method="multitaper", band=(8, 13)
        )
        metric = np.average(metric)  # average across selected channels

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
        fill_perc = np.clip((metric - min_) / (max_ - min_), 0, 1)

        # update feedback
        feedback.fill_perc = fill_perc
        feedback.show()

    # close the feedback window
    feedback.close()
