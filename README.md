[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# NeuroFeedback demo

This repository can be installed via `pip` with `pip install git+https://github.com/mscheltienne/demo-nfb`.

It implements a simple neurofeedback loop of 30 seconds using [BSL](https://bsl-tools.github.io/).

# Example

```
from demo_nfb import basic

stream_name = ...  # str
basic(stream_name)
```
