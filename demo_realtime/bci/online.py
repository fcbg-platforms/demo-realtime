from __future__ import annotations  # c.f. PEP 563, PEP 649

import time
from typing import TYPE_CHECKING

import numpy as np
from mne import make_fixed_length_epochs
from mne.io import RawArray
from mne_lsl.stream import StreamLSL as Stream

from ..utils._checks import check_type
from ..utils._docs import fill_doc
from ..utils._imports import import_optional_dependency
from ..utils.logs import logger
from ..visuals import CarGame

if TYPE_CHECKING:
    from tensorflow.keras.models import Model


@fill_doc
def online(stream_name: str, model: Model, duration: int = 60) -> None:
    """Run the online BCI-game.

    Parameters
    ----------
    %(stream_name)s
    model : Model
        Fitted EEGNet model.
    %(duration)s
    """
    import_optional_dependency("tensorflow")

    from tensorflow.keras.models import Model

    check_type(stream_name, (str,), "stream_name")
    check_type(model, (Model,), "model")
    check_type(duration, ("int-like",), "duration")
    assert 0 < duration

    # create receiver and feedback
    stream = Stream(bufsize=2.0, name=stream_name).connect()
    stream.drop_channels(["X1", "X2", "X3", "A2", "TRG"])
    stream.set_montage("standard_1020")
    game = CarGame()

    # wait to fill one buffer
    time.sleep(2.0)

    try:
        game.start()
        time.sleep(2.0)
        start = time.time()
        while time.time() - start <= duration:
            # retrieve data
            data, _ = stream.get_data()
            raw = RawArray(data, stream.info)
            raw.filter(
                l_freq=2.0,
                h_freq=25.0,
                method="fir",
                phase="zero-double",
                fir_window="hamming",
                fir_design="firwin",
                pad="edge",
            )
            raw.resample(128)  # shape is now (20, 256)
            epochs = make_fixed_length_epochs(raw, duration=1.0, overlap=0.9)

            # retrieve numpy array and transform to NHWC
            # (n_trials, n_channels, n_samples, n_kernels)
            X = epochs.get_data() * 1000
            X = X.reshape(*X.shape, 1)  # n_kernels = 1

            # predict
            pred = model(X, training=False).numpy().argmax(axis=-1)
            # apply a running mean to smooth the predictions
            N = 3  # window size for the running mean
            pred = np.convolve(pred, np.ones(N) / N, mode="valid")
            # we should have 9 prediction values based on the number of epochs and on
            # the convolution parameters, let's assume that an action is requested if
            # the 2 last predictions are identical and if they are integers (i.e. not
            # part of a transition between 2 states).
            if all(p.is_integer() for p in pred[-2:]) and pred[-1] == pred[-2]:
                pred = pred[-1]
                logger.info("Predicting %i", pred)
            else:
                logger.debug("Predictions after smoothing: %s", pred)
                logger.info("No new prediction.")
                continue

            # do an action based on the prediction
            if pred == 0:  # turn left
                logger.debug("Prediction: going left.")
                game.go_direction("left")
            elif pred == 1:
                logger.debug("Prediction: going right.")
                game.go_direction("right")
            elif pred == 2:
                logger.debug("Prediction: going straight.")
                pass

    except Exception:
        raise
    finally:
        game.stop()
        del stream
