from .metrics import bandpower as bandpower
from .utils._checks import check_type as check_type
from .utils._docs import fill_doc as fill_doc
from .utils.logs import verbose as verbose
from .visuals import DoubleSpinningWheel as DoubleSpinningWheel

def nfb_double_spinning_wheel(
    stream_name: str,
    winsize: float = 3,
    duration: float = 30,
    *,
    verbose: bool | str | int | None = None,
) -> None:
    """Real-time neurofeedback loop using a double spinning wheel as feedback.

    The feedback represents the alpha-band power on the occipital electrodes
    ``O1`` and ``O2``.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    winsize : float
        Duration of an acquisition window.
    duration : float
        Duration of the real-time loop.
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.
    """
