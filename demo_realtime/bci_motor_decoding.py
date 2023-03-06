import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import numpy as np
from bsl import StreamRecorder
from bsl.triggers import SoftwareTrigger

from .utils._checks import _check_type, _ensure_path
from .visuals._bci_motor_decoding import Calibration


def offline_calibration(
    n_repetition: int,
    stream_name: str,
    directory: Optional[Union[str, Path]] = None,
):
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
    stream_name : str
        Name of the amplifier LSL stream.
    directory : path-like | None
        Path where the dataset is recorded. If None, a temporary directory is
        used.
    """
    _check_type(n_repetition, ("int",), "n_repetition")
    assert 0 < n_repetition  # sanity-check
    _check_type(stream_name, (str,), "stream_name")
    if directory is None:
        tempdir = TemporaryDirectory(prefix="tmp_demo-realtime_")
        directory = Path(tempdir.name)
    directory = _ensure_path(directory, must_exist=True)

    # generate random cue order -- 1: left fist, 2: right fist, 3: hands open
    cues = [1] * n_repetition + [2] * n_repetition + [3] * n_repetition
    rng = np.random.default_rng()
    rng.shuffle(cues)

    # the current recorder and the associated software trigger will be
    # deprecated in version 1.0 in favor of a safer LabRecorder-based approach.
    recorder = StreamRecorder(directory, stream_name=stream_name)
    recorder.start()
    trigger = SoftwareTrigger(recorder)

    try:
        # create psychopy window and objects
        window = Calibration(
            size=(1920, 1080), screen=1, fullscr=True, allowGUI=False
        )
        window.show_instructions()
        window.show_examples()
        window.cross.setAutoDraw(True)
        time.sleep(2)

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
            time.sleep(2.5)

            # remove cue
            if cue == 1:  # left fist
                window.lfist.setAutoDraw(False)
            elif cue == 2:  # right fist
                window.rfist.setAutoDraw(False)
            elif cue == 3:  # hands open
                window.lhand.setAutoDraw(False)
                window.rhand.setAutoDraw(False)
            window.window.flip()
            time.sleep(1.5)

    except Exception:
        raise
    finally:
        window.close()
        # save the file
        trigger.close()
        recorder.stop()
