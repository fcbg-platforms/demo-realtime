import cv2

from ..utils._docs import fill_doc
from ._visual import _Visual


@fill_doc
class Cross(_Visual):
    """
    Class to display a cross, e.g. a fixation cross.

    Parameters
    ----------
    %(window_name)s
    %(window_size)s
    """

    def __init__(self, window_name="Visual", window_size=None):
        super().__init__(window_name, window_size)

    def putCross(self, length, thickness, color, position):
        """
        Draw a cross composed of 2 rectangles defined by length and thickness.
        The rectangles are positionned to form a cross by overlapping.

        - Horizontal rectangle
        P1 --------------
        |                |
         -------------- P2

        - Vertical rectangle
        P1 --
        |    |
        |    |
        |    |
        |    |
        |    |
         -- P2

        Parameters
        ----------
        length : int
            Number of pixels used to draw the length of the cross.
        thickness : int
            Number of pixels used to draw the thickness of the cross.
        color : str | tuple
            Color used to fill the cross as a matplotlib string or a (B, G, R)
            tuple of int8 set between 0 and 255.
        position : str | list
            Position of the center of the cross. Either the string 'center' or
            'centered' to position the cross in the center of the window; or a
            2-length sequence of positive integer defining the position of the
            center of the cross in the window. The position is defined in
            opencv coordinates, with (0, 0) being the top left corner of the
            window.
        """
        length = Cross._check_length(length, self.window_size)
        thickness = Cross._check_thickness(thickness, length)
        color = _Visual._check_color(color)
        position = Cross._check_position(
            position, length, self.window_size, self.window_center
        )

        # Horizontal rectangle
        xP1 = position[0] - length // 2
        yP1 = position[1] - thickness // 2
        xP2 = xP1 + length
        yP2 = yP1 + thickness
        cv2.rectangle(self._img, (xP1, yP1), (xP2, yP2), color, -1)

        # Vertical rectangle
        xP1 = position[0] - thickness // 2
        yP1 = position[1] - length // 2
        xP2 = xP1 + thickness
        yP2 = yP1 + length
        cv2.rectangle(self._img, (xP1, yP1), (xP2, yP2), color, -1)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_length(length, window_size):
        """
        Checks that the length is a strictly positive integer shorter than the
        width or the height of the window.
        """
        length = int(length)
        assert 0 < length
        assert all(length <= size for size in window_size)
        return length

    @staticmethod
    def _check_thickness(thickness, length):
        """
        Checks that the thickness is a strictly positive integer shorter than
        length.
        """
        thickness = int(thickness)
        assert 0 < thickness
        assert thickness < length
        return thickness

    @staticmethod
    def _check_position(position, length, window_size, window_center):
        """
        Checks that the inputted position of the center of the cross allows
        the cross to fit in the window.
        The position is given as (X, Y) in opencv coordinates, with (0, 0)
        being the top left corner of the window.
        """
        if isinstance(position, str):
            position = position.lower().strip()
            assert position in ["centered", "center"]
            position = window_center
        position = tuple(position)
        assert len(position) == 2
        assert 0 <= position[0] - length // 2
        assert position[0] - length // 2 + length <= window_size[0]
        assert 0 <= position[1] - length // 2
        assert position[1] - length // 2 + length <= window_size[1]
        return position
