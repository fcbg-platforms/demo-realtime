from abc import ABC, abstractmethod
from typing import Any

from _typeshed import Incomplete
from matplotlib import pyplot as plt
from mne import Info
from numpy.typing import NDArray

from ..utils._checks import check_type as check_type
from ..utils._docs import copy_doc as copy_doc
from ..utils._docs import fill_doc as fill_doc
from ..utils.logs import logger as logger

class _BaseTopomap(ABC):
    """Abstract class defining a topographic map feedback.

    Parameters
    ----------
    info : Info
        MNE Info instance with a montage.
    """

    _info: Incomplete
    _vmin: Incomplete
    _vmax: Incomplete
    _inc: int
    _vmin_arr: Incomplete
    _vmax_arr: Incomplete

    @abstractmethod
    def __init__(self, info: Info): ...
    @abstractmethod
    def update(self, topodata: NDArray[float]) -> None:
        """Update the topographic map with the new data array.

        Parameters
        ----------
        topodata : array of shape (n_channels,)
            1D array of shape (n_channels,) containing the new data samples to
            plot.
        """

    @abstractmethod
    def close(self):
        """Close the topographic map feedback window."""

    @property
    def info(self) -> Info:
        """MNE Info instance with a montage.

        :type: `mne.Info`
        """

    @property
    def vlim(self) -> tuple[float, float]:
        """Colormap range."""

    @staticmethod
    def _check_info(info: Any) -> None:
        """Check that the info instance has a montage."""

class TopomapMPL(_BaseTopomap):
    """Topographic map feedback using matplotlib.

    Parameters
    ----------
    info : Info
        MNE Info instance with a montage.
    cmap : str
        The matplotlib color map name.
    figsize : tuple
        2-sequence tuple defining the matplotlib figure size as (width, height)
        in inches.
    """

    _cmap: Incomplete
    _kwargs: Incomplete

    def __init__(
        self,
        info: Info,
        cmap: str = "Purples",
        figsize: tuple[float, float] | list[float] | None = None,
    ) -> None: ...
    def update(self, topodata: NDArray[float]):
        """Update the topographic map with the new data array.

        Parameters
        ----------
        topodata : array of shape (n_channels,)
            1D array of shape (n_channels,) containing the new data samples to
            plot.
        """

    def close(self) -> None:
        """Close the topographic map feedback window."""

    @property
    def fig(self) -> plt.Figure:
        """Matplotlib figure."""

    @property
    def axes(self) -> plt.Axes:
        """Matplotlib axes."""

    @property
    def cmap(self) -> str:
        """Matplotlib colormap name."""

    @staticmethod
    def _check_figsize(figsize: Any) -> tuple[float, float]:
        """Check the figure size."""
