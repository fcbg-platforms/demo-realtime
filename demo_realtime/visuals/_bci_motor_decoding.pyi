from _typeshed import Incomplete
from psychopy.visual import ImageStim, ShapeStim, Window

from ..utils._imports import import_optional_dependency as import_optional_dependency

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

    _window: Incomplete
    _keyboard: Incomplete
    _lfist: Incomplete
    _rfist: Incomplete
    _lhand: Incomplete
    _rhand: Incomplete
    _cross: Incomplete

    def __init__(self, **kwargs) -> None: ...
    def show_instructions(self) -> None:
        """Display instructions on the window."""

    def show_examples(self) -> None:
        """Display examples on the window."""

    def close(self) -> None:
        """Close the visual window."""

    def __del__(self) -> None:
        """Make sure to close the window and clear the events before del."""

    @property
    def window(self) -> Window:
        """The PsychoPy window."""

    @property
    def lfist(self) -> ImageStim:
        """The left fist PsychoPy image."""

    @property
    def rfist(self) -> ImageStim:
        """The right fist PsychoPy image."""

    @property
    def lhand(self) -> ImageStim:
        """The left hand PsychoPy image."""

    @property
    def rhand(self) -> ImageStim:
        """The left hand PsychoPy image."""

    @property
    def cross(self) -> ShapeStim:
        """The PsychoPy shape used as a fixation cross."""
