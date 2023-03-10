# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from bsl import StreamReceiver, StreamRecorder
from bsl.triggers import SoftwareTrigger
from mne import Epochs, find_events
from mne.io import read_raw_fif

from .utils._checks import _check_type, _ensure_path
from .utils._docs import fill_doc
from .utils._import import _import_optional_dependency
from .utils._logs import logger
from .visuals import CarGame
from .visuals._bci_motor_decoding import Calibration

if TYPE_CHECKING:
    from tensorflow.keras import Model


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
    _check_type(n_repetition, ("int",), "n_repetition")
    assert 0 < n_repetition  # sanity-check
    _check_type(stream_name, (str,), "stream_name")
    directory = _ensure_path(directory, must_exist=True)

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
        window = Calibration(
            size=(1920, 1080), screen=1, fullscr=True, allowGUI=False
        )
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
        # property recorder.fname in versions prior to 1.0.
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
    fname: Union[str, Path], directory: Union[str, Path] = None
) -> Union[Model, Path]:
    """Fit EEGNet model with the dataset recorded with offline_calibration.

    Parameters
    ----------
    fname : path-like
        Path to the FIFF recording.
    directory : path-like
        Path where the model checkpoints are saved.

    Returns
    -------
    model : Model
        Fitted EEGNet model.
    fname : Path
        Path to the H5 file containing the weights.
    """
    _import_optional_dependency("tensorflow")

    from tensorflow.keras import utils as np_utils
    from tensorflow.keras.callbacks import ModelCheckpoint

    from ._bci_EEGNet import EEGNet

    fname = _ensure_path(fname, must_exist=True)
    directory = _ensure_path(directory, must_exist=True)

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
    epochs = Epochs(
        raw, events, event_id, tmin=0, tmax=1, baseline=None, preload=True
    )
    epochs.resample(128)

    # extract raw data and labels. The raw data is scaled by 1000 due to
    # scaling sensitivity in deep learning.
    labels = epochs.events[:, -1]
    X = epochs.get_data() * 1000  # shape is (n_trials, n_channels, n_samples)
    Y = labels

    del raw
    del epochs

    # split the dataset into train/validate/test with equal number of labels
    # in each split.
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
    idx_train = np.hstack(
        (lfist_idx[:n2], rfist_idx[:n2], hands_open_idx[:n2])
    )
    X_train = X[idx_train, :, :]
    Y_train = Y[idx_train]

    n1 = n2
    n2 = n1 + int(0.15 * size)
    idx_val = np.hstack(
        (lfist_idx[n1:n2], rfist_idx[n1:n2], hands_open_idx[n1:n2])
    )
    X_validate = X[idx_val, :, :]
    Y_validate = Y[idx_val]

    n1 = n2
    idx_test = np.hstack((lfist_idx[n1:], rfist_idx[n1:], hands_open_idx[n1:]))
    X_test = X[idx_test, :, :]
    Y_test = Y[idx_test]

    del n1
    del n2

    # convert labels to one-hot encodings.
    Y_train = np_utils.to_categorical(Y_train - 1)
    Y_validate = np_utils.to_categorical(Y_validate - 1)
    Y_test = np_utils.to_categorical(Y_test - 1)

    # convert data to NHWC (n_trials, n_channels, n_samples, n_kernels) format.
    n_channels, n_samples, n_kernels = 20, 128, 1
    X_train = X_train.reshape(
        X_train.shape[0], n_channels, n_samples, n_kernels
    )
    X_validate = X_validate.reshape(
        X_validate.shape[0], n_channels, n_samples, n_kernels
    )
    X_test = X_test.reshape(X_test.shape[0], n_channels, n_samples, n_kernels)

    # create and fit model
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
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    logger.info("Number of parameters: %i", model.count_params())

    # set a valid path for your system to record model checkpoints
    fname_chkpoint = (
        directory / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime())}.h5"
    )
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

    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    logger.info("Classification accuracy [test]: %f", acc)

    return model, fname_chkpoint


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
    _check_type(stream_name, (str,), "stream_name")
    _check_type(duration, ("int"), "duration")
    assert 0 < duration

    # create receiver and feedback
    sr = StreamReceiver(bufsize=1.0, winsize=1.0, stream_name=stream_name)
    sr.mne_infos[stream_name].set_montage("standard_1020", on_missing="ignore")
    game = CarGame()

    # wait to fill one buffer
    sr.acquire()
    time.sleep(1.0)

    try:
        game.start()
        start = time.time()
        while time.time() - start <= duration:
            # retrieve data
            sr.acquire()
            raw, _ = sr.get_window(return_raw=True)
            raw.drop_channels(["X1", "X2", "X3", "A2"])
            raw.filter(
                l_freq=2.0,
                h_freq=25.0,
                method="fir",
                phase="zero-double",
                fir_window="hamming",
                fir_design="firwin",
                pad="edge",
            )
            raw.resample(128)  # shape is now (20, 128)

            # retrieve numpy array and transform to NHWC
            # (n_trials, n_channels, n_samples, n_kernels)
            X = raw.get_data() * 1000
            X = X.reshape(1, X.shape[0], X.shape[1], 1)

            # predict
            prob = model.predict(X)
            pred = prob.argmax(axis=-1)[0]
            logger.info("Predicting %i", pred)

            # do an action based on the prediction
            if pred == 0:  # turn left
                game.go_left()
            elif pred == 1:
                game.go_right()  # turn right
            elif pred == 2:
                pass

    except Exception:
        raise
    finally:
        game.stop()
        del sr
