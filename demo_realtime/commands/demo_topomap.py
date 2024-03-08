import argparse

from mne import set_log_level as mne_set_log_level
from mne_lsl import set_log_level as mne_lsl_set_log_level
from mne_lsl.lsl import resolve_streams

from .. import rt_topomap


def run():
    """Run 'demo-topomap' command."""
    mne_lsl_set_log_level("INFO")
    mne_set_log_level("INFO")

    parser = argparse.ArgumentParser(
        prog="demo-topomap",
        description="Start a demo of real-time topographic map.",
    )
    parser.add_argument(
        "-s",
        "--stream_name",
        type=str,
        metavar="str",
        help="stream to connect to.",
    )
    parser.add_argument(
        "-w",
        "--winsize",
        type=float,
        metavar="float",
        help="duration of the acquisition window (seconds).",
        default=3.0,
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        metavar="float",
        help="duration of the nfb loop (seconds).",
        default=30.0,
    )
    parser.add_argument(
        "--figsize",
        type=float,
        metavar="float",
        nargs=2,
        help="figure size for the matplotlib backend.",
    )
    parser.add_argument("--verbose", help="enable debug logs.", action="store_true")
    args = parser.parse_args()

    stream_name = args.stream_name
    if stream_name is None:
        streams = [stream.name for stream in resolve_streams() if stream.stype == "EEG"]
        assert len(streams) == 1
        stream_name = streams[0]
    rt_topomap(
        stream_name,
        args.winsize,
        args.duration,
        args.figsize,
        verbose="DEBUG" if args.verbose else "INFO",
    )
