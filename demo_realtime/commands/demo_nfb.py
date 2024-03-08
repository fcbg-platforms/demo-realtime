import argparse

from mne_lsl import set_log_level as mne_lsl_set_log_level
from mne_lsl.lsl import resolve_streams

from .. import nfb_double_spinning_wheel, nfb_filling_bar


def run():
    """Run 'demo-nfb' command."""
    mne_lsl_set_log_level("INFO")

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
        "--bar", help="use a filling bar feedback.", action="store_true"
    )
    parser.add_argument(
        "--wheel",
        help="use a double spinning wheel feedback.",
        action="store_true",
    )
    parser.add_argument("--verbose", help="enable debug logs.", action="store_true")
    args = parser.parse_args()

    if (args.bar and args.wheel) or (not args.bar and not args.wheel):
        raise RuntimeError(
            "One and only one of the flag '--bar' or '--wheel' must be provided to "
            "chose between the 2 possible feedbacks."
        )

    stream_name = args.stream_name
    if stream_name is None:
        streams = [stream.name for stream in resolve_streams() if stream.stype == "EEG"]
        assert len(streams) == 1
        stream_name = streams[0]

    if args.bar:
        function = nfb_filling_bar
    elif args.wheel:
        function = nfb_double_spinning_wheel

    function(
        stream_name,
        args.winsize,
        args.duration,
        verbose="DEBUG" if args.verbose else "INFO",
    )
