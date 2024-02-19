from pathlib import Path

from demo_realtime import set_log_level
from demo_realtime.bci import calibration, fit_EEGNet, online

set_log_level("INFO")

directory = Path.home() / "Downloads" / "bci"
fname = calibration(10, "WS-default", directory)
model = fit_EEGNet(fname, "WS-default")
online("WS-default", model, duration=300)
