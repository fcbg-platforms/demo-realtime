[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![codecov](https://codecov.io/gh/fcbg-platforms/demo-realtime/graph/badge.svg?token=EN5L5ZS6HG)](https://codecov.io/gh/fcbg-platforms/demo-realtime)
[![tests](https://github.com/fcbg-platforms/demo-realtime/actions/workflows/pytest.yaml/badge.svg?branch=main)](https://github.com/fcbg-platforms/demo-realtime/actions/workflows/pytest.yaml)

# Real-time demo using LSL

This repository can be installed via `pip` with `pip install git+https://github.com/fcbg-platforms/demo-realtime`.

It implements simple real-time demonstration using [MNE-LSL](https://mne.tools/mne-lsl).

# Example

```
from demo_realtime import nfb_filling_bar

stream_name: str = "My LSL Stream"
nfb_filling_bar(stream_name)
```
