from __future__ import annotations  # c.f. PEP 563, PEP 649

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from bsl import StreamRecorder
from bsl.triggers import SoftwareTrigger
from mne import Epochs, concatenate_epochs, find_events, make_fixed_length_epochs
from mne.io import RawArray, read_raw_fif
from mne_lsl.stream import StreamLSL as Stream
from scipy.stats import mode

from .utils._checks import check_type, ensure_path
from .utils._docs import fill_doc
from .utils._imports import import_optional_dependency
from .utils.logs import logger
from .visuals import CarGame
from .visuals._bci_motor_decoding import Calibration

if TYPE_CHECKING:
    from typing import Optional, Union

    from tensorflow.keras.models import Model


def offline_calibration(
    n_repetition: int,
    stream_name: str,
    directory: Union[str, Path] = None,
) -> Path:
    """Gather a dataset of training and validation epochs.

    3 actions (class) are recorded in a randomized order:
        - rest with the hands on the table
        - clench the left fist
        - clench the right fist

    Parameters
    ----------
    n_repetition : int
        Number of repetition of each of the 3 actions. e.g. ``10`` will set the
        calibration to measure 10 epochs of each class.
    %(stream_name)s
    directory : path-like
        Path where the dataset is recorded.

    Returns
    -------
    fname : Path
        Path to the FIFF recording.
    """
    check_type(n_repetition, ("int-like",), "n_repetition")
    assert 0 < n_repetition  # sanity-check
    check_type(stream_name, (str,), "stream_name")
    directory = ensure_path(directory, must_exist=True)

    # generate random cue order -- 1: left fist, 2: right fist, 3: hands open
    cues = [1] * n_repetition + [2] * n_repetition + [3] * n_repetition
    rng = np.random.default_rng()
    rng.shuffle(cues)

    # the current recorder and the associated software trigger will be
    # deprecated in version 1.0 in favor of a safer LabRecorder-based approach.
    recorder = StreamRecorder(directory, stream_name=stream_name)
    recorder.start()
    trigger = SoftwareTrigger(recorder)

    try:
        # create psychopy window and objects
        window = Calibration(size=(1920, 1080), screen=1, fullscr=True, allowGUI=False)
        window.show_instructions()
        window.show_examples()
        window.cross.setAutoDraw(True)
        time.sleep(2)

        # loop until all cues are exhausted
        while len(cues) != 0:
            # handle new cue
            cue = cues.pop(0)
            if cue == 1:  # left fist
                window.lfist.setAutoDraw(True)
            elif cue == 2:  # right fist
                window.rfist.setAutoDraw(True)
            elif cue == 3:  # hands open
                window.lhand.setAutoDraw(True)
                window.rhand.setAutoDraw(True)
            window.window.flip()
            trigger.signal(cue)
            time.sleep(2.5)

            # remove cue
            if cue == 1:  # left fist
                window.lfist.setAutoDraw(False)
            elif cue == 2:  # right fist
                window.rfist.setAutoDraw(False)
            elif cue == 3:  # hands open
                window.lhand.setAutoDraw(False)
                window.rhand.setAutoDraw(False)
            window.window.flip()
            time.sleep(1.5)

        # reconstruct fname from the 'eve_file' because BSL does not update the
        # property recorder.fname
        fname = (
            directory
            / "fif"
            / recorder.eve_file.name.replace(
                "-eve.txt",
                f"-{recorder.stream_name}-raw.fif",
            )
        )

    except Exception:
        raise
    finally:
        window.close()
        # save the file
        trigger.close()
        recorder.stop()

    return fname


def offline_fit(
    fname: Union[str, Path],
    model: Optional[Union[Model, str, Path]] = None,
) -> Model:
    """Fit EEGNet model with the dataset recorded with offline_calibration.

    Parameters
    ----------
    fname : path-like
        Path to the FIFF recording.
    model : Model | path-like | None
        If provided, model on which fit is resumed. If None, a new model is created.

    Returns
    -------
    model : Model
        Fitted EEGNet model.
    """
    import_optional_dependency("tensorflow")

    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.utils import to_categorical

    from ._bci_EEGNet import EEGNet

    fname = ensure_path(fname, must_exist=True)
    check_type(model, (Model, str, Path, None), "model")
    if isinstance(model, (str, Path)):
        model = ensure_path(model, must_exist=True)
        model = load_model(model)

    # load and preprocess dataset
    raw = read_raw_fif(fname, preload=False)
    raw.drop_channels(["X1", "X2", "X3", "A2"])
    raw.load_data()
    raw.filter(
        l_freq=2.0,
        h_freq=25.0,
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge",
    )
    raw.set_montage("standard_1020")

    # create epochs and resample to 128 Hz
    events = find_events(raw, stim_channel="TRIGGER")
    event_id = dict(lfist=1, rfist=2, hands_open=3)
    epochs_list = list()
    for step in np.arange(0, 1.1, 0.1):
        epochs = Epochs(
            raw,
            events,
            event_id,
            tmin=0.5 + step,
            tmax=1.5 + step,
            baseline=None,
            preload=True,
        )
        epochs.shift_time(tshift=0, relative=False)
        epochs_list.append(epochs)
    epochs = concatenate_epochs(epochs_list)
    epochs.resample(128)
    del raw

    # extract raw data and labels. The raw data is scaled by 1000 due to scaling
    # sensitivity in deep learning.
    labels = epochs.events[:, -1]
    X = epochs.get_data() * 1000  # shape is (n_trials, n_channels, n_samples)
    Y = labels

    # split the dataset into train/validate/test with equal number of labels in each
    # split.
    lfist_idx = np.where(Y == event_id["lfist"])[0]
    rfist_idx = np.where(Y == event_id["rfist"])[0]
    hands_open_idx = np.where(Y == event_id["hands_open"])[0]
    assert lfist_idx.size == rfist_idx.size == hands_open_idx.size
    size = lfist_idx.size

    rng = np.random.default_rng()
    rng.shuffle(lfist_idx)
    rng.shuffle(rfist_idx)
    rng.shuffle(hands_open_idx)

    n2 = int(0.7 * size)
    idx_train = np.hstack((lfist_idx[:n2], rfist_idx[:n2], hands_open_idx[:n2]))
    rng.shuffle(idx_train)
    X_train = X[idx_train, :, :]
    Y_train = Y[idx_train]

    n1 = n2
    n2 = n1 + int(0.15 * size)
    idx_val = np.hstack((lfist_idx[n1:n2], rfist_idx[n1:n2], hands_open_idx[n1:n2]))
    rng.shuffle(idx_val)
    X_validate = X[idx_val, :, :]
    Y_validate = Y[idx_val]

    n1 = n2
    idx_test = np.hstack((lfist_idx[n1:], rfist_idx[n1:], hands_open_idx[n1:]))
    rng.shuffle(idx_test)
    X_test = X[idx_test, :, :]
    Y_test = Y[idx_test]

    del n1
    del n2

    # convert labels to one-hot encodings.
    Y_train = to_categorical(Y_train - 1)
    Y_validate = to_categorical(Y_validate - 1)
    Y_test = to_categorical(Y_test - 1)

    # convert data to NHWC (n_trials, n_channels, n_samples, n_kernels) format.
    n_channels, n_samples, n_kernels = (
        epochs.info["nchan"],
        epochs.times.size,
        1,
    )
    X_train = X_train.reshape(X_train.shape[0], n_channels, n_samples, n_kernels)
    X_validate = X_validate.reshape(
        X_validate.shape[0], n_channels, n_samples, n_kernels
    )
    X_test = X_test.reshape(X_test.shape[0], n_channels, n_samples, n_kernels)

    # create and fit model
    if model is None:
        model = EEGNet(
            len(event_id),
            n_channels,
            n_samples,
            dropoutRate=0.5,
            kernelLength=32,
            F1=8,
            D=2,
            F2=16,
            dropoutType="Dropout",
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        logger.info("Number of parameters: %i", model.count_params())

    # set a valid path for your system to record model checkpoints
    tempdir = TemporaryDirectory(prefix="tmp_demo-realtime_")
    fname_chkpoint = tempdir / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime())}.h5"
    checkpointer = ModelCheckpoint(
        filepath=fname_chkpoint,
        verbose=1,
        save_best_only=True,
    )
    model.fit(
        X_train,
        Y_train,
        batch_size=16,
        epochs=300,
        verbose=2,
        validation_data=(X_validate, Y_validate),
        callbacks=[checkpointer],
    )

    # load optimal weights
    model.load_weights(fname_chkpoint)
    del tempdir

    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    logger.info("Classification accuracy [test]: %f", acc)

    return model


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
    stream.drop_channels(["X1", "X2", "X3", "A2"])
    stream.set_montage("standard_1020", on_missing="ignore")
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
            prob = model(X, training=False)
            pred = mode(prob.numpy().argmax(axis=-1), keepdims=False)[0]
            logger.info("Predicting %i", pred)

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
            time.sleep(0.5)

    except Exception:
        raise
    finally:
        game.stop()
        del stream
