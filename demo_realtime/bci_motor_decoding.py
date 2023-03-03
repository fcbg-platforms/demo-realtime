from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

from bsl import StreamRecorder
from bsl.triggers import SoftwareTrigger

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

from .utils._checks import _check_type, _ensure_path
from .utils._imports import _import_optional_dependency

# psychopy settings
_SCREEN_SIZE = (1920, 1080)
_WINTYPE = "pyglet"
_UNITS = "norm"
_SCREEN = 1
_FULL_SCREEN = True
_ALLOW_GUI = False


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
    _import_optional_dependency("psychopy")

    from psychopy.hardware.keyboard import Keyboard
    from psychopy.visual import ImageStim, TextStim, Window

    _check_type(n_repetition, ("int",), "n_repetition")
    assert 0 < n_repetition  # sanity-check
    _check_type(stream_name, (str,), "stream_name")
    if directory is None:
        tempdir = TemporaryDirectory(prefix="tmp_demo-realtime_")
        directory = Path(tempdir.name)
    directory = _ensure_path(directory, must_exist=True)

    # the current recorder and the associated software trigger will be
    # deprecated in version 1.0 in favor of a safer LabRecorder-based approach.
    recorder = StreamRecorder(directory, stream_name=stream_name)
    recorder.start()
    trigger = SoftwareTrigger(recorder)

    # create psychopy window and objects
    window = Window(
        size=_SCREEN_SIZE,
        winType=_WINTYPE,
        screen=_SCREEN,
        fullscr=_FULL_SCREEN,
        allowGUI=_ALLOW_GUI,
        color=(0, 0, 0),
    )

    keyboard = Keyboard()
    window.callOnFlip(keyboard.clearEvents, eventType="keyboard")
    keyboard.stop()
    keyboard.clearEvents()

    image = files("demo_realtime.feedbacks") / "resources" / "fist-clench.png"
    assert image.is_file() and image.suffix == ".png"  # sanity-check
    lfist = ImageStim(
        window, image=image, size=[0.5, 0.5], pos=[-0.7, 0], ori=-20
    )
    rfist = ImageStim(
        window, image=image, size=[-0.5, 0.5], pos=[0.7, 0], ori=20
    )
    image = files("demo_realtime.feedbacks") / "resources" / "hand-open.png"
    assert image.is_file() and image.suffix == ".png"  # sanity-check
    lhand = ImageStim(
        window, image=image, size=[0.6, 0.6], pos=[-0.7, 0], ori=20
    )
    rhand = ImageStim(
        window, image=image, size=[-0.6, 0.6], pos=[0.7, 0], ori=-20
    )

    instructions = TextStim(
        win=window,
        text="Welcome to the game calibration!\n\n"
        "You will have 3 commands to repeat to calibrate the game:\n"
        "- Clench the left fist\n- Clench the right fist\n"
        "- Keep both hands open\n\n"
        "The cues are randomize.\n You might have to perform the same "
        "action several times in a row.\n\n"
        "To start an example, press SPACE.",
        height=0.04,
        pos=(0, 0.05),
    )
    instructions.setAutoDraw(True)
    window.flip()

    # wait for SPACE
    keyboard.start()
    while True:  # wait for 'space'
        keys = keyboard.getKeys(keyList=["space"], waitRelease=False)
        if len(keys) != 0:
            break
        window.flip()
    keyboard.stop()
    keyboard.clearEvents()

    # save the file
    trigger.close()
    recorder.stop()
