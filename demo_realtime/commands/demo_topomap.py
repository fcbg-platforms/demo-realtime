import argparse

from bsl import set_log_level as bsl_set_log_level
from bsl.utils.lsl import search_lsl

from .. import rt_topomap


def run():
    """Run 'demo-topomap' command."""
    bsl_set_log_level("INFO")

    parser = argparse.ArgumentParser(
        prog="demo_nfb",
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
    args = parser.parse_args()
    stream_name = args.stream_name
    if stream_name is None:
        stream_name = search_lsl(ignore_markers=True, timeout=3)
    rt_topomap(stream_name, args.winsize, args.duration)
