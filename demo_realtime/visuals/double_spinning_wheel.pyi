from multiprocessing import Value
from pathlib import Path

from _typeshed import Incomplete
from numpy.typing import NDArray

from ..utils._checks import check_type as check_type
from ..utils._imports import import_optional_dependency as import_optional_dependency

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

    _winkwargs: Incomplete
    _image: Incomplete
    _wheel_size: Incomplete
    _offset: Incomplete
    _speed: Incomplete
    _status: Incomplete
    _process: Incomplete

    def __init__(
        self, wheel_size: float = 0.4, offset: float = 0.5, **kwargs
    ) -> None: ...
    def start(self) -> None:
        """Start the visual feedback."""

    def stop(self) -> None:
        """Stop the visual feedback."""

    def __del__(self) -> None:
        """Make sure to stop the feedback and close the window before del."""

    @staticmethod
    def _main_loop(
        winkargs: dict,
        image: Path,
        wheel_size: float,
        offset: float,
        speed: Value,
        status: Value,
    ) -> None: ...
    @staticmethod
    def _normalize_size(winsize: NDArray[int], wheel_size: float) -> NDArray[float]:
        """Normalize the size to retain the aspect ratio of the image.

        Parameters
        ----------
        winsize : array of shape (2,)
            Size of the PsychoPy window.
        wheel_size : float
            Normalized size of the image, between -1 and 1.
        """

    @property
    def image(self) -> Path:
        """Path to the image of the wheel displayed.

        :type: :class:`~pathlib.Path`
        """

    @property
    def offset(self) -> float:
        """Normalized offset of the images.

        :type: :class:`float`
        """

    @property
    def wheel_size(self) -> float:
        """Normalized size of the wheel images.

        :type: :class:`float`
        """

    @property
    def speed(self) -> int:
        """Speed of the rotation. Typical values range from 0 to 100.

        :type: :class:`int`
        """

    @speed.setter
    def speed(self, speed: int) -> None:
        """Speed of the rotation. Typical values range from 0 to 100.

        :type: :class:`int`
        """

    @property
    def active(self) -> bool:
        """Return True if the feedback is running.

        :type: :classl:`bool`
        """
