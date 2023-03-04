# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

from ..utils._imports import _import_optional_dependency
from ..utils._logs import logger

if TYPE_CHECKING:
    from psychopy.visual import ImageStim, ShapeStim, Window


class Calibration:
    """PsychoPy visuals used to calibrate the BCI motor decoding example.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments are provided to `psychopy.visual.Window`.
        The pre-defined values are:
        * ``units='norm'``
        * ``winType="pyglet"``
        * ``color=(0, 0, 0)``
    """

    def __init__(self, **kwargs) -> None:
        _import_optional_dependency("psychopy")

        from psychopy.hardware.keyboard import Keyboard
        from psychopy.visual import ImageStim, ShapeStim, Window

        # prepare psychopy settings
        if "units" not in kwargs:
            kwargs["units"] = "norm"
        elif kwargs["units"] != "norm":
            raise ValueError(
                f"The unit used should be 'norm'. Provided {kwargs['units']} "
                "is not supported."
            )
        if "winType" not in kwargs:
            kwargs["winType"] = "pyglet"
        elif kwargs["winType"] != "pyglet":
            logger.warning(
                "The 'pyglet' window type is recommended above the provided "
                "'%s'.",
                kwargs["winType"],
            )

        if "color" not in kwargs:
            kwargs["color"] = (0, 0, 0)
        elif kwargs["color"] != (0, 0, 0):
            logger.warning(
                "The color '(0, 0, 0)' is recommended above the provided "
                "'%s'.",
                kwargs["color"],
            )

        # prepare psychopy window and objects
        self._window = Window(**kwargs)
        self._keyboard = Keyboard()
        self._window.callOnFlip(
            self._keyboard.clearEvents, eventType="keyboard"
        )
        self._keyboard.stop()
        self._keyboard.clearEvents()

        image = (
            files("demo_realtime.visuals") / "resources" / "fist-clench.png"
        )
        assert image.is_file() and image.suffix == ".png"  # sanity-check
        self._lfist = ImageStim(
            self._window, image=image, size=[0.5, 0.5], pos=[-0.7, 0], ori=-20
        )
        self._rfist = ImageStim(
            self._window, image=image, size=[-0.5, 0.5], pos=[0.7, 0], ori=20
        )
        image = files("demo_realtime.visuals") / "resources" / "hand-open.png"
        assert image.is_file() and image.suffix == ".png"  # sanity-check
        self._lhand = ImageStim(
            self._window, image=image, size=[0.6, 0.6], pos=[-0.7, 0], ori=20
        )
        self._rhand = ImageStim(
            self._window, image=image, size=[-0.6, 0.6], pos=[0.7, 0], ori=-20
        )
        self._cross = ShapeStim(
            win=self._window,
            vertices="cross",
            units="height",
            size=(0.05, 0.05),
            lineColor="white",
            fillColor="white",
        )

    def show_instructions(self) -> None:
        """Display instructions on the window."""
        from psychopy.visual import TextStim

        instructions = TextStim(
            win=self._window,
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
        self._window.flip()

        # wait for SPACE
        self._keyboard.start()
        while True:  # wait for 'space'
            keys = self._keyboard.getKeys(keyList=["space"], waitRelease=False)
            if len(keys) != 0:
                break
            self._window.flip()
        self._keyboard.stop()
        self._keyboard.clearEvents()
        instructions.setAutoDraw(False)
        self._window.flip()

    def show_examples(self) -> None:
        """Display examples on the window."""
        from psychopy.visual import TextStim

        # left fist
        instructions = TextStim(
            win=self._window,
            text="This is a 'clench the left fist' cue.",
            height=0.04,
            pos=(0.2, 0.05),
        )
        instructions.setAutoDraw(True)
        self._lfist.setAutoDraw(True)
        self._window.flip()
        time.sleep(2)
        instructions.setAutoDraw(False)
        self._lfist.setAutoDraw(False)

        # inter-trial
        instructions = TextStim(
            win=self._window,
            text="Break between the cues.\n"
            "Keep your hands open as for the 'hands open' cue.",
            height=0.04,
            pos=(0.0, 0.05),
        )
        instructions.setAutoDraw(True)
        self._cross.setAutoDraw(True)
        time.sleep(2)
        instructions.setAutoDraw(False)
        self._cross.setAutoDraw(False)

        # hands open
        instructions = TextStim(
            win=self._window,
            text="This is a 'keep the hands open' cue.",
            height=0.04,
            pos=(0.0, 0.05),
        )
        instructions.setAutoDraw(True)
        self._lhand.setAutoDraw(True)
        self._rhand.setAutoDraw(True)
        self._window.flip()
        time.sleep(2)
        instructions.setAutoDraw(False)
        self._lhand.setAutoDraw(False)
        self._rhand.setAutoDraw(False)

        # inter-trial
        instructions = TextStim(
            win=self._window,
            text="Break between the cues.\n"
            "Keep your hands open as for the 'hands open' cue.",
            height=0.04,
            pos=(0.0, 0.05),
        )
        instructions.setAutoDraw(True)
        self._cross.setAutoDraw(True)
        time.sleep(2)
        instructions.setAutoDraw(False)
        self._cross.setAutoDraw(False)

        # right fist
        instructions = TextStim(
            win=self._window,
            text="This is a 'clench the right fist' cue.",
            height=0.04,
            pos=(-0.2, 0.05),
        )
        instructions.setAutoDraw(True)
        self._rfist.setAutoDraw(True)
        self._window.flip()
        time.sleep(2)
        instructions.setAutoDraw(False)
        self._rfist.setAutoDraw(False)
        self._window.flip()

    # -------------------------------------------------------------------------
    @property
    def window(self) -> Window:
        """The PsychoPy window."""
        return self._window

    @property
    def lfist(self) -> ImageStim:
        """The left fist PsychoPy image."""
        return self._lfist

    @property
    def rfist(self) -> ImageStim:
        """The right fist PsychoPy image."""
        return self._rfist

    @property
    def lhand(self) -> ImageStim:
        """The left hand PsychoPy image."""
        return self._lhand

    @property
    def rhand(self) -> ImageStim:
        """The left hand PsychoPy image."""
        return self._rhand

    @property
    def cross(self) -> ShapeStim:
        """The PsychoPy shape used as a fixation cross."""
        return self._cross
