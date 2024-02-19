import argparse
from tempfile import TemporaryDirectory

from mne_lsl.lsl import resolve_streams

from .. import set_log_level
from ..bci import calibration, fit_EEGNet, online


def run():
    """Run 'demo-bci' command."""
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
    set_log_level("DEBUG" if args.verbose else "INFO")
    stream_name = args.stream_name
    if stream_name is None:
        streams = resolve_streams(timeout=3)
        streams = [stream for stream in streams if stream.sfreq != 0]
        if len(streams) != 1:
            raise RuntimeError(
                "Multiple streams found. Please provide the stream name explicitly."
            )
        stream_name = streams[0].name

    directory = TemporaryDirectory(prefix="tmp_demo-realtime_")
    fname = calibration(10, stream_name, directory.name)
    input(">>> Press ENTER to continue..")
    model = fit_EEGNet(fname)
    input(">>> Press ENTER to continue..")
    online("WS-default", model, duration=args.duration)
