from __future__ import annotations  # c.f. PEP 563, PEP 649

import time
from typing import TYPE_CHECKING

from mne_lsl.stream import StreamLSL as Stream

from .metrics import bandpower
from .utils._checks import check_type
from .utils._docs import fill_doc
from .utils.logs import verbose
from .visuals import TopomapMPL

if TYPE_CHECKING:
    from typing import Optional, Union


@fill_doc
@verbose
def rt_topomap(
    stream_name: str,
    winsize: float = 3,
    duration: float = 30,
    figsize: Optional[tuple[float, float]] = None,
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
    # check inputs
    check_type(stream_name, (str,), "stream_name")
    check_type(winsize, ("numeric",), "winsize")
    assert 0 < winsize
    check_type(duration, ("numeric",), "duration")
    assert 0 < duration

    # create receiver and feedback
    stream = Stream(bufsize=winsize, name=stream_name).connect()
    stream.drop_channels(("TRIGGER", "TRG", "X1", "X2", "X3", "A1", "A2"))
    stream.set_montage("standard_1020")

    # create feedback
    feedback = TopomapMPL(stream.info, "Purples", figsize)

    # wait to fill one buffer
    time.sleep(winsize)

    # main loop
    start = time.time()
    while time.time() - start <= duration:
        data, _ = stream.get_data()
        # compute metric
        metric = bandpower(
            data, fs=stream.info["sfreq"], method="periodogram", band=(8, 13)
        )
        # update feedback
        feedback.update(metric)

    # close the feedback window
    feedback.close()
