from .metrics import bandpower as bandpower
from .utils._checks import check_type as check_type
from .utils._docs import fill_doc as fill_doc
from .utils.logs import verbose as verbose
from .visuals import TopomapMPL as TopomapMPL

def rt_topomap(
    stream_name: str,
    winsize: float = 3,
    duration: float = 30,
    figsize: tuple[float, float] | None = None,
    *,
    verbose: bool | str | int | None = None,
):
    """Real-time topographic feedback loop.

    The feedback represents the alpha-band relative power measured by a DSI-24
    amplifier.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    winsize : float
        Duration of an acquisition window.
    duration : float
        Duration of the real-time loop.
    figsize : tuple
        2-sequence tuple defining the matplotlib figure size as (width, height)
        in inches.
    verbose : int | str | bool | None
        Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
        ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
        verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
        ``"WARNING"`` for False and to ``"INFO"`` for True.
    """
