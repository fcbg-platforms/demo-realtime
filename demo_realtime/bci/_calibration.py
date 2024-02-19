from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np
from bsl import StreamRecorder
from bsl.triggers import SoftwareTrigger

from ..utils._checks import check_type, ensure_path
from ..utils._docs import fill_doc
from ..utils._imports import import_optional_dependency
from ..visuals._bci_motor_decoding import Calibration

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
def calibration(
    n_repetition: int,
    stream_name: str,
    directory: str | Path = None,
) -> Path:
    """Gather a dataset of training and validation epochs.

    3 actions (class) are recorded in a randomized order:
        - rest with the hands on the table
        - clench the left fist
        - clench the right fist

    Parameters
    ----------
    n_repetition : int
        Number of repetition of each of the 3 actions. e.g. ``10`` will set the
        calibration to measure 10 epochs of each class.
    %(stream_name)s
    directory : path-like
        Path where the dataset is recorded.

    Returns
    -------
    fname : Path
        Path to the FIFF recording.
    """
    import_optional_dependency("psychopy")

    from psychopy.core import wait

    check_type(n_repetition, ("int-like",), "n_repetition")
    assert 0 < n_repetition  # sanity-check
    check_type(stream_name, (str,), "stream_name")
    directory = ensure_path(directory, must_exist=True)

    # generate random cue order -- 1: left fist, 2: right fist, 3: hands open
    cues = [1] * n_repetition + [2] * n_repetition + [3] * n_repetition
    rng = np.random.default_rng()
    rng.shuffle(cues)
    # create recorder and trigger
    recorder = StreamRecorder(directory, stream_name=stream_name)
    recorder.start()
    trigger = SoftwareTrigger(recorder)
    # create psychopy window
    window = Calibration(size=(1920, 1080), screen=1, fullscr=True, allowGUI=False)

    try:
        # create psychopy window and objects
        window.show_instructions()
        window.show_examples()
        window.cross.setAutoDraw(True)
        wait(2)

        # loop until all cues are exhausted
        while len(cues) != 0:
            # handle new cue
            cue = cues.pop(0)
            if cue == 1:  # left fist
                window.lfist.setAutoDraw(True)
            elif cue == 2:  # right fist
                window.rfist.setAutoDraw(True)
            elif cue == 3:  # hands open
                window.lhand.setAutoDraw(True)
                window.rhand.setAutoDraw(True)
            window.window.flip()
            trigger.signal(cue)
            wait(2.5)

            # remove cue
            if cue == 1:  # left fist
                window.lfist.setAutoDraw(False)
            elif cue == 2:  # right fist
                window.rfist.setAutoDraw(False)
            elif cue == 3:  # hands open
                window.lhand.setAutoDraw(False)
                window.rhand.setAutoDraw(False)
            window.window.flip()
            wait(1.5)

        # reconstruct fname from the 'eve_file' because BSL does not update the
        # property recorder.fname
        fname = (
            directory
            / "fif"
            / recorder.eve_file.name.replace(
                "-eve.txt",
                f"-{recorder.stream_name}-raw.fif",
            )
        )

    except Exception:
        raise
    finally:
        window.close()
        # save the file
        trigger.close()
        recorder.stop()

    return fname
