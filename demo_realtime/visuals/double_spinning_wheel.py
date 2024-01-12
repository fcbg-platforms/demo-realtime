from importlib.resources import files
from multiprocessing import Process, Value
from pathlib import Path
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from ..utils._checks import check_type
from ..utils._imports import import_optional_dependency


class DoubleSpinningWheel:
    """Feedback visual using 2 counter-spinning wheel.

    The rotation speed can be adjusted in real-time with the ``speed`` property which
    typically ranges from 0 to 100.

    Parameters
    ----------
    wheel_size : float
        Normalized size of the wheel image. The provided value will be converted to
        retain the aspect ratio of the image.
    offset : float
        Normalized offset to position the image on the left and right side of the
        screen.
    **kwargs : dict
        Additional keyword arguments are provided to :class:`psychopy.visual.Window`.
        The pre-defined values are:
        * ``units='norm'``
        * ``winType="pyglet"``
        * ``color=(-1, -1, -1)``
    """

    def __init__(
        self,
        wheel_size: float = 0.4,
        offset: float = 0.5,
        **kwargs,
    ) -> None:
        import_optional_dependency("psychopy")
        # prepare psychopy settings
        if "units" not in kwargs:
            kwargs["units"] = "norm"
        elif kwargs["units"] != "norm":
            raise ValueError(
                f"The unit used should be 'norm'. Provided {kwargs['units']} is not "
                "supported."
            )
        if "winType" not in kwargs:
            kwargs["winType"] = "pyglet"
        elif kwargs["winType"] != "pyglet":
            warn(
                "The 'pyglet' window type is recommended above the provided "
                f"'{kwargs['winType']}'.",
                RuntimeWarning,
                stacklevel=2,
            )
        if "color" not in kwargs:
            kwargs["color"] = (-1, -1, -1)
        elif kwargs["color"] != (-1, -1, -1):
            warn(
                "The color '(-1, -1, -1)' is recommended above the provided "
                f"'{kwargs['color']}'.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._winkwargs = kwargs

        # store image path
        image = files("demo_realtime.visuals") / "resources" / "wheel.png"
        assert image.is_file() and image.suffix == ".png"  # sanity-check
        self._image = image
        # and image settings
        check_type(wheel_size, ("numeric",), "wheel_size")
        check_type(offset, ("numeric",), "offset")
        for var, name in [(wheel_size, "wheel_size"), (offset, "offset")]:
            if var < -1 or var > 1:
                warn(
                    f"Normalized {name} should be in the range (-1, 1). Values outside "
                    "this range might yield an image outside of the window.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._wheel_size = wheel_size
        self._offset = offset

        # prepare shared variables and process to control the wheel
        self._speed = Value("i", 0)
        self._status = Value("i", 0)
        self._process = Process(
            target=DoubleSpinningWheel._main_loop,
            args=(
                self._winkwargs,
                self._image,
                self._wheel_size,
                self._offset,
                self._speed,
                self._status,
            ),
        )

    def start(self) -> None:
        """Start the visual feedback."""
        if self._status.value == 1:
            assert self._process.is_alive()
            raise RuntimeError("The feedback is already started.")
        with self._status.get_lock():
            self._status.value = 1
        self._process.start()

    def stop(self) -> None:
        """Stop the visual feedback."""
        if self._status.value == 0:
            assert not self._process.is_alive()
            raise RuntimeError("The feedback is already stopped.")
        with self._status.get_lock():
            self._status.value = 0
        self._process.join(5)
        if self._process.is_alive():  # sanity-check
            warn(
                "The feedback process did not stop properly.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._process.kill()

    def __del__(self):
        """Make sure to stop the feedback and close the window before del."""
        if self._status.value == 1:
            self.stop()

    # -------------------------------------------------------------------------
    @staticmethod
    def _main_loop(
        winkargs: dict,
        image: Path,
        wheel_size: float,
        offset: float,
        speed: Value,
        status: Value,
    ) -> None:
        from psychopy.visual import ImageStim, Window

        # open window
        win = Window(**winkargs)
        # normalize the image size to retain the aspect ratio
        wheel_size = DoubleSpinningWheel._normalize_size(win.size, wheel_size)
        lwheel = ImageStim(win, image=image, size=wheel_size * [1, 1], pos=[-offset, 0])
        rwheel = ImageStim(win, image=image, size=wheel_size * [-1, 1], pos=[offset, 0])
        lwheel.autoDraw = True
        rwheel.autoDraw = True
        win.flip()

        # run infinite display-loop
        while True:
            if status.value == 0:
                break

            # assuming speed set between [1, 100]
            lwheel.ori += speed.value / 10
            rwheel.ori -= speed.value / 10
            lwheel.draw()
            rwheel.draw()
            win.flip()

        # close window after a stop is requested
        win.close()

    @staticmethod
    def _normalize_size(
        winsize: NDArray[int],
        wheel_size: float,
    ) -> NDArray[float]:
        """Normalize the size to retain the aspect ratio of the image.

        Parameters
        ----------
        winsize : array of shape (2,)
            Size of the PsychoPy window.
        wheel_size : float
            Normalized size of the image, between -1 and 1.
        """
        if winsize[0] == winsize[1]:
            wheel_size = (wheel_size, wheel_size)
        elif winsize[1] < winsize[0]:
            wheel_size = (wheel_size, wheel_size * winsize[0] / winsize[1])
        elif winsize[0] < winsize[1]:
            wheel_size = (wheel_size * winsize[1] / winsize[0], wheel_size)
        return np.array(wheel_size)

    # -------------------------------------------------------------------------
    @property
    def image(self) -> Path:
        """Path to the image of the wheel displayed.

        :type: :class:`~pathlib.Path`"""
        return self._image

    @property
    def offset(self) -> float:
        """Normalized offset of the images.

        :type: :class:`float`
        """
        return self._offset

    @property
    def wheel_size(self) -> float:
        """Normalized size of the wheel images.

        :type: :class:`float`
        """
        return self._wheel_size

    @property
    def speed(self) -> int:
        """Speed of the rotation. Typical values range from 0 to 100.

        :type: :class:`int`
        """
        return self._speed.value

    @speed.setter
    def speed(self, speed: int) -> None:
        """Setter used to change the rotation speed."""
        assert speed == int(speed), "The provided speed must be an integer."
        with self._speed.get_lock():
            self._speed.value = speed

    @property
    def active(self) -> bool:
        """Return True if the feedback is running.

        :type: :classl:`bool`
        """
        return bool(self._status.value)
