import cv2

from ..utils._docs import fill_doc
from ._visual import _Visual


@fill_doc
class Text(_Visual):
    """
    Class to display a text.

    Parameters
    ----------
    %(window_name)s
    %(window_size)s
    """

    def __init__(self, window_name="Visual", window_size=None):
        super().__init__(window_name, window_size)

    def putText(
        self,
        text,
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=2,
        color="white",
        thickness=2,
        position="centered",
    ):
        """
        Method adding text to the visual.

        Parameters
        ----------
        text : str
            Text to display.
        fontFace : cv2 font
            Font to use to write the text.
        fontScale : int
            Scale of the font.
        color : str | tuple
            Color used to write the text as a matplotlib string or a (B, G, R)
            tuple of int8 set between 0 and 255.
        thickness : int
            Text line thickness in pixel.
        position : str | list
            Position of the bottom left corner of the text. Either the string
            'center' or 'centered' to position the text in the center of the
            window; or a 2-length sequence of positive integer defining the
            position of the bottom left corner of the text in the window. The
            position is defined in opencv coordinates, with (0, 0) being the
            top left corner of the window.\n
        """
        if text != "":
            textWidth, textHeight = cv2.getTextSize(
                text, fontFace, fontScale, thickness
            )[0]
            position = Text._check_position(
                position,
                textWidth,
                textHeight,
                self.window_size,
                self.window_center,
            )
            color = _Visual._check_color(color)

            cv2.putText(
                self._img,
                text,
                position,
                fontFace,
                fontScale,
                color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    # --------------------------------------------------------------------
    @staticmethod
    def _check_position(
        position, textWidth, textHeight, window_size, window_center
    ):
        """
        Checks that the inputted position of the bottom left corner of the
        text allows the text to fit in the window.
        The position is given as (X, Y) in opencv coordinates, with (0, 0)
        being the top left corner of the window.
        """
        if isinstance(position, str):
            position = position.lower().strip()
            assert position in ["centered", "center"]
            position = (
                window_center[0] - textWidth // 2,
                window_center[1] + textHeight // 2,
            )

        position = tuple(position)
        assert len(position) == 2
        assert 0 <= position[0]
        assert position[0] + textWidth <= window_size[0]
        assert 0 <= position[1] - textHeight
        assert position[1] <= window_size[1]
        return position
