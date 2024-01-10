from __future__ import annotations  # c.f. PEP 563, PEP 649

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from mne import Info
from mne.viz import plot_topomap

from ..utils._checks import check_type
from ..utils._docs import copy_doc, fill_doc
from ..utils.logs import logger

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


@fill_doc
class _BaseTopomap(ABC):
    """Abstract class defining a topographic map feedback.

    Parameters
    ----------
    %(info)s
    """

    @abstractmethod
    def __init__(self, info: Info) -> None:
        _BaseTopomap._check_info(info)
        self._info = info
        # variables used to define and update the colorbar range
        self._vmin = None
        self._vmax = None
        self._inc = 0
        self._vmin_arr = np.ones(100) * np.nan
        self._vmax_arr = np.ones(100) * np.nan

    @abstractmethod
    def update(self, topodata: NDArray[float]) -> None:
        """Update the topographic map with the new data array.

        Parameters
        ----------
        topodata : array of shape (n_channels,)
            1D array of shape (n_channels,) containing the new data samples to
            plot.
        """
        # update arrays that stores 100 points for vmin/vmax
        self._vmin_arr[self._inc % 100] = np.min(topodata)
        self._vmax_arr[self._inc % 100] = np.max(topodata)
        self._inc += 1
        # log when 100 points have passed
        if self._inc == 100:
            logger.info("Vmin/Vmax calibrated!")
        # update vmin/vmax
        self._vmin = np.percentile(self._vmin_arr[~np.isnan(self._vmin_arr)], 5)
        self._vmax = np.percentile(self._vmax_arr[~np.isnan(self._vmax_arr)], 95)
        logger.debug("%i --Vmin: %.3f -- Vmax: %.3f", self._inc, self._vmin, self._vmax)

    @abstractmethod
    def close(self):
        """Close the topographic map feedback window."""
        pass

    # ------------------------------------------------------------------------
    @property
    def info(self) -> Info:
        """MNE Info instance with a montage.

        :type: `mne.Info`
        """
        return self._info

    @property
    def vlim(self) -> tuple[float, float]:
        """Colormap range."""
        return (self._vmin, self._vmax)

    # ------------------------------------------------------------------------
    @staticmethod
    def _check_info(info: Any) -> None:
        """Check that the info instance has a montage."""
        check_type(info, (Info,), "info")
        if info.get_montage() is None:
            raise ValueError(
                "The provided info instance 'info' does not have "
                "a DigMontage attached."
            )


@fill_doc
class TopomapMPL(_BaseTopomap):
    """Topographic map feedback using matplotlib.

    Parameters
    ----------
    %(info)s
    cmap : str
        The matplotlib color map name.
    %(figsize)s
    """

    def __init__(
        self,
        info: Info,
        cmap: str = "Purples",
        figsize: tuple[float, float] = (3, 3),
    ) -> None:
        if plt.get_backend() != "QtAgg":
            plt.switch_backend("QtAgg")
        if not plt.isinteractive():
            plt.ion()  # enable interactive mode
        super().__init__(info)
        check_type(cmap, (str,), "cmap")
        self._cmap = cmap
        figsize = TopomapMPL._check_figsize(figsize)
        self._fig, self._axes = plt.subplots(1, 1, figsize=figsize)
        # define kwargs for plot_topomap
        self._kwargs = dict(
            cmap=self._cmap,
            sensors=True,
            res=64,
            axes=self._axes,
            names=None,
            outlines="head",
            contours=0,
            onselect=None,
            extrapolate="auto",
            show=False,
        )
        # create initial topographic plot
        plot_topomap(
            np.zeros(len(self._info["ch_names"])),
            self._info,
            **self._kwargs,
        )

    @copy_doc(_BaseTopomap.update)
    def update(self, topodata: NDArray[float]):
        super().update(topodata)
        self._axes.clear()
        plot_topomap(
            topodata,
            self._info,
            vlim=self.vlim,
            **self._kwargs,
        )
        # redraw the canvas
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    @copy_doc(_BaseTopomap.close)
    def close(self):
        plt.close(self._fig)

    # ------------------------------------------------------------------------
    @property
    def fig(self) -> plt.Figure:
        """Matplotlib figure."""
        return self._fig

    @property
    def axes(self) -> plt.Axes:
        """Matplotlib axes."""
        return self._axes

    @property
    def cmap(self) -> str:
        """Matplotlib colormap name."""
        return self._cmap

    # ------------------------------------------------------------------------
    @staticmethod
    def _check_figsize(figsize: Any) -> tuple[float, float]:
        """Check the figure size."""
        check_type(figsize, (tuple, list), "figsize")
        if len(figsize) != 2:
            raise ValueError(
                "The figure size should be a 2-item tuple "
                "defining the matplotlib figure size (width, "
                "height) in inches."
            )
        for elt in figsize:
            check_type(elt, ("numeric",))
        if any(elt <= 0 for elt in figsize):
            raise ValueError(
                "The figure size should be a 2-item tuple of "
                "strictly positive numbers."
            )
        if figsize[0] != figsize[1]:
            logger.warning(
                "Topographic maps are best displayed in a square "
                "axes, define with a figsize (width, height) with "
                "width = height."
            )
        if any(5 < elt for elt in figsize):
            logger.warning(
                "Large figsize will increase the render time and "
                "slow down the online loop, which can create "
                "stuttering."
            )
        return tuple(figsize)
