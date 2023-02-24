from __future__ import annotations

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

from typing import TYPE_CHECKING
from multiprocessing import Process, Value

import numpy as np
from numpy.typing import NDArray

from ..utils._imports import import_optional_dependency

if TYPE_CHECKING:
    from psychopy.visual import Window, ImageStim


class DoubleSpinningWheel:
    """Feedback visual using 2 counter-spinning wheel.

    Parameters
    ----------
    size : float
        Normalized size of the wheel image. The provided value will be
        converted to retain the aspect ratio of the image.
    offset : float
        Normalized offset to position the image on the left and right side of
        the screen.
    **kwargs : dict
        Additional keyword arguments are provided to `psychopy.visual.Window`.
        The already defined values are:
        * ``units='norm'``
        * ``winType="pyglet"``
        * ``color=(-1, -1, -1)``
    """

    def __init__(
        self,
        size: float = 0.4,
        offset: float = 0.5,
        **kwargs,
    ) -> None:
        import_optional_dependency("psychopy")
        from psychopy.visual import Window, ImageStim

        # prepare PsychoPy objects
        self._win = Window(
            units="norm", winType="pyglet", color=(-1, -1, -1), **kwargs
        )
        fname = files("demo_realtime.feedbacks").joinpath(
            "resources/wheel.png"
        )

        size = DoubleSpinningWheel._normalize_size(self._win.size, size)
        self._lwheel = ImageStim(
            self._win, image=fname, size=size, pos=[-offset, 0]
        )
        self._rwheel = ImageStim(
            self._win, image=fname, size=size, pos=[offset, 0]
        )
        self._lwheel.autoDraw = True
        self._rwheel.autoDraw = True
        self._win.flip()

        # prepare shared variables and process to control the wheel
        self._speed = Value("i", 0)
        self._status = Value("i", 0)  # 0: stop feedback, 1: run feedback
        self._process = Process(
            target=DoubleSpinningWheel._update_window,
            args=(
                self._speed,
                self._status,
                self._win,
                self._lwheel,
                self._rwheel,
            ),
        )

    def start(self) -> None:
        """Start the feedback visual."""
        if self._status.value == 1:
            raise RuntimeError("The feedback is already started.")

        self._lwheel.draw()
        self._rwheel.draw()
        self._win.flip()

        with self._status.get_lock():
            self._status.value = 1
        self._process.start()

    def stop(self) -> None:
        """Stop the feedback visual."""
        if self._status.value == 0:
            raise RuntimeError("The feedback is already stopped.")
        with self._status.get_lock():
            self._status.value = 0
        self._process.join(2)
        if self._process.is_alive():
            self._process.kill()

    def close(self) -> None:
        """Close the feedback window."""
        self._win.close()

    def __del__(self):
        """Make sure to stop the feedback and close the window before del."""
        if self._status.value == 1:
            self.stop()
        self.close()

    # -------------------------------------------------------------------------
    @staticmethod
    def _update_window(
        speed: Value,
        status: Value,
        win: Window,
        lwheel: ImageStim,
        rwheel: ImageStim,
    ) -> None:
        while True:
            if status.value == 0:
                break

            lwheel.ori += speed.value
            rwheel.ori += speed.value
            lwheel.draw()
            rwheel.draw()
            win.flip()

    @staticmethod
    def _normalize_size(winsize: NDArray[int], size: float) -> NDArray[float]:
        """Normalize the size to retain the aspect ratio of the image.

        Parameters
        ----------
        winsize : array of shape (2,)
            Size of the PsychoPy window.
        size : float
            Normalized size of the image, between -1 and 1.
        """
        if winsize[0] == winsize[1]:
            size = (size, size)
        elif winsize[1] < winsize[0]:
            size = (size, size * winsize[0] / winsize[1])
        elif winsize[0] < winsize[1]:
            size = (size * winsize[1] / winsize[0], size)
        return np.array(size)

    # -------------------------------------------------------------------------
    @property
    def win(self) -> Window:
        """PsychoPy window object."""
        return self._win

    @property
    def lwheel(self) -> ImageStim:
        """PsychoPy imagestim object of the left wheel."""
        return self._lwheel

    @property
    def rwheel(self) -> ImageStim:
        """PsychoPy imagestim object of the right wheel."""
        return self._rwheel

    @property
    def speed(self) -> int:
        """Speed of the rotation.

        :type: int
        """
        return self._speed.value

    @speed.setter
    def speed(self, speed: int) -> None:
        """Setter used to change the rotation speed."""
        assert speed == int(speed), "The provided speed must be an integer."
        with self._speed.get_lock():
            self._speed.value = speed

    @property
    def running(self) -> bool:
        """Return True if the feedback is running.

        :type: bool
        """
        return bool(self._status.value)
