from pathlib import Path

from ..utils._checks import check_type as check_type
from ..utils._checks import ensure_path as ensure_path
from ..utils._docs import fill_doc as fill_doc
from ..utils._imports import import_optional_dependency as import_optional_dependency
from ..visuals._bci_motor_decoding import Calibration as Calibration
from ._config import EVENT_ID as EVENT_ID

def calibration(
    n_repetition: int,
    stream_name: str,
    directory: str | Path = None,
    *,
    skip_instructions: bool = False,
) -> Path:
    """Gather a dataset of training and validation epochs.

    3 actions (class) are recorded in a randomized order:
        - rest with the hands on the table
        - clench the left fist
        - clench the right fist

    Parameters
    ----------
    n_repetition : int
        Number of repetition of each of the 3 actions. e.g. ``10`` will set the
        calibration to measure 10 epochs of each class.
    stream_name : str
        The name of the LSL stream to connect to.
    directory : path-like
        Path where the dataset is recorded.
    skip_instructions : bool
        If True, instructions and examples are skipped.

    Returns
    -------
    fname : Path
        Path to the FIFF recording.
    """
