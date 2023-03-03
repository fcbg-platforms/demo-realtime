import time

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

from ..utils._imports import _import_optional_dependency
from ..utils._logs import logger


class Calibration:
    def __init__(self, **kwargs):
        _import_optional_dependency("psychopy")

        from psychopy.hardware.keyboard import Keyboard
        from psychopy.visual import ImageStim, Window

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

    def show_instructions(self):
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

    def show_examples(self):
        from psychopy.visual import TextStim

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
