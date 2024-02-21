from tensorflow.keras.models import Model

from ..utils._checks import check_type as check_type
from ..utils._docs import fill_doc as fill_doc
from ..utils._imports import import_optional_dependency as import_optional_dependency
from ..utils.logs import logger as logger
from ..visuals import CarGame as CarGame

def online(stream_name: str, model: Model, duration: int = 60) -> None:
    """Run the online BCI-game.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    model : Model
        Fitted EEGNet model.
    duration : float
        Duration of the real-time loop.
    """
