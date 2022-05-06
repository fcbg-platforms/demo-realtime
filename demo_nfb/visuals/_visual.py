from abc import ABC, abstractmethod

import cv2
from matplotlib import colors
import numpy as np
import screeninfo

from ..utils._docs import fill_doc


@fill_doc
class _Visual(ABC):
    """
    Base visual class.

    Parameters
    ----------
    %(window_name)s
    %(window_size)s
    """

    @abstractmethod
    def __init__(self, window_name='Visual', window_size=None):
        self._window_name = str(window_name)

        # Size attributes
        self._window_size = _Visual._check_window_size(window_size)
        self._window_width = self._window_size[0]
        self._window_height = self._window_size[1]
        self._window_center = (self._window_width//2, self._window_height//2)

        # Default black background
        self._img = np.full(
            (self._window_height, self._window_width, 3),
            fill_value=(0, 0, 0), dtype=np.uint8)
        self._background = [0, 0, 0]

    def show(self, wait=1):
        """
        Show the visual with cv2.imshow() and cv2.waitKey().

        Parameters
        ----------
        wait : int
            Wait timer passed to cv2.waitKey() [ms].
        """
        cv2.imshow(self._window_name, self._img)
        cv2.waitKey(wait)

    def close(self):
        """
        Close the visual.
        """
        cv2.destroyWindow(self._window_name)

    def draw_background(self, color):
        """
        Draw a uniform single color background. Replaces all the pixels with
        this color, thus erasing any prior work.

        Parameters
        ----------
        color : str | tuple
            Color used to draw the background as a matplotlib string or a
            (B, G, R) tuple of int8 set between 0 and 255.
        """
        color = _Visual._check_color(color)
        self._img = np.full(
            (self._window_height, self._window_width, 3),
            fill_value=color, dtype=np.uint8)
        self._background = color

    def __del__(self):
        """Close when deleting the object."""
        self.close()

    # --------------------------------------------------------------------
    @staticmethod
    def _check_window_size(window_size):
        """
        Checks if the window size is valid or set it as the minimum
        (width, height) supported by any connected monitor.
        """
        if window_size is not None:
            window_size = tuple(int(size) for size in window_size)
            assert len(window_size) == 2
            assert all(0 < size for size in window_size)
        else:
            try:
                width = min(
                    monitor.width for monitor in screeninfo.get_monitors())
                height = min(
                    monitor.height for monitor in screeninfo.get_monitors())
            except ValueError as headless:
                raise ValueError from headless
            window_size = (width, height)
        return window_size

    @staticmethod
    def _check_color(color):
        """
        Checks if a color is valid and converts it to BGR.
        """
        if isinstance(color, str):
            r, g, b, _ = colors.to_rgba(color)
            color = [int(c*255) for c in (b, g, r)]
        elif isinstance(color, (tuple, list)) and len(color) == 3:
            color = [int(c) for c in color]
            assert all(0 <= c <= 255 for c in color)
        else:
            raise TypeError
        return color

    @staticmethod
    def _check_axis(axis):
        """
        Checks that the axis is valid and converts it to integer (0, 1).
            - 0: Vertical
            - 1: Horizontal
        """
        if isinstance(axis, str):
            axis = axis.lower().strip()
            assert axis in ['horizontal', 'h', 'vertical', 'v']
            axis = 0 if axis.startswith('v') else 1
        elif isinstance(axis, (bool, int, float)):
            axis = int(axis)
            assert axis in (0, 1)
        else:
            raise TypeError
        return axis

    # --------------------------------------------------------------------
    @property
    def window_name(self):
        """
        Window's name.
        """
        return self._window_name

    @property
    def window_size(self):
        """
        Window's size (width x height).
        """
        return self._window_size

    @property
    def window_center(self):
        """
        Window's center position.
        """
        return self._window_center

    @property
    def img(self):
        """
        Image array.
        """
        return self._img

    @property
    def background(self):
        """
        Background color.
        """
        return self._background

    @background.setter
    def background(self, background):
        self.draw_background(background)
