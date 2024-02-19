import argparse
from tempfile import TemporaryDirectory

from bsl import set_log_level as bsl_set_log_level
from bsl.utils.lsl import search_lsl

from ..bci import calibration, fit_EEGNet, online


def run():
    """Run 'demo-bci' command."""
    bsl_set_log_level("INFO")

    parser = argparse.ArgumentParser(
        prog="demo-nfb",
        description="Start a demo of a neurofeedback system.",
    )
    parser.add_argument(
        "-s",
        "--stream_name",
        type=str,
        metavar="str",
        help="stream to connect to.",
    )
    parser.add_argument(
        "-n",
        "--n_repetition",
        type=int,
        metavar="int",
        help="number of repetitions in the calibration.",
        default=10,
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        metavar="float",
        help="duration of the game loop (seconds).",
        default=300,
    )
    parser.add_argument("--verbose", help="enable debug logs.", action="store_true")
    args = parser.parse_args()

    stream_name = args.stream_name
    if stream_name is None:
        stream_name = search_lsl(ignore_markers=True, timeout=3)

    directory = TemporaryDirectory(prefix="tmp_demo-realtime_")
    fname = calibration(10, stream_name, directory)
    input(">>> Press ENTER to continue..")
    model = fit_EEGNet(fname, stream_name)
    input(">>> Press ENTER to continue..")
    online("WS-default", model, duration=args.duration)
