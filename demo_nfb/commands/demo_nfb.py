import argparse

from bsl import set_log_level as bsl_set_log_level
from bsl.utils.lsl import search_lsl

from .. import basic


def run():
    """Run demo_nfb() command."""
    bsl_set_log_level("INFO")

    parser = argparse.ArgumentParser(
        prog="demo_nfb",
        description="Start a demo of basic neurofeedback.",
    )
    parser.add_argument(
        "-s",
        "--stream_name",
        type=str,
        metavar="str",
        help="stream to connect to.",
    )
    args = parser.parse_args()
    stream_name = args.stream_name
    if stream_name is None:
        stream_name = search_lsl(ignore_markers=True, timeout=3)
    basic(stream_name)
