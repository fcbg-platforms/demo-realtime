from __future__ import annotations  # c.f. PEP 563, PEP 649

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from mne import Epochs, concatenate_epochs, find_events
from mne.io import read_raw_fif
from mne.preprocessing import compute_current_source_density

from ..utils._checks import check_type, ensure_path
from ..utils.logs import logger
from ._config import EVENT_ID

if TYPE_CHECKING:
    from mne import BaseEpochs
    from numpy.typing import NDArray
    from tensorflow.keras.models import Model


def _load_dataset(fname: str | Path) -> BaseEpochs:
    """Load and preprocess the EEG dataset from the DSI-24 amplifier.

    Parameters
    ----------
    fname : path-like
        Path to the FIFF recording.

    Returns
    -------
    epochs : Epochs
        Epochs preprocessed for the 2 events, 'lfist' and 'rfist'.
    """
    fname = ensure_path(fname, must_exist=True)
    raw = read_raw_fif(fname, preload=False)
    raw.pick(["TRIGGER", "P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "Pz"])
    raw.load_data()
    raw.set_montage("standard_1020")
    raw.filter(
        l_freq=2.0,
        h_freq=25.0,
        method="iir",  # 4th order butterworth
        phase="forward",  # causal IIR filter
    )
    # create epochs and resample to 128 Hz
    events = find_events(raw, stim_channel="TRIGGER")
    epochs_list = list()
    for step in np.arange(0, 1.1, 0.1):
        epochs = Epochs(
            raw,
            events,
            EVENT_ID,
            picks="eeg",
            tmin=0.5 + step,
            tmax=1.5 + step,
            baseline=None,
            preload=True,
        )
        epochs.shift_time(tshift=0, relative=False)
        epochs_list.append(epochs)
    epochs = concatenate_epochs(epochs_list)
    epochs.resample(128)
    epochs = compute_current_source_density(epochs)
    return epochs


def _get_data(
    epochs: BaseEpochs,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Extract labels and dataset from the epochs.

    Parameters
    ----------
    epochs : Epochs
        Epochs preprocessed for the 2 events, 'lfist' and 'rfist'.

    Returns
    -------
    X_train : array of shape (n_trials, n_channels, n_samples)
        Training dataset.
    Y_train : array of shape (n_trials,)
        Training labels.
    X_validate : array of shape (n_trials, n_channels, n_samples)
        Validation dataset.
    Y_validate : array of shape (n_trials,)
        Validation labels.
    X_test : array of shape (n_trials, n_channels, n_samples)
        Test dataset.
    Y_test : array of shape (n_trials,)
        Test labels.
    """
    # the raw data is scaled by 1000 due to scaling sensitivity in deep learning
    X = epochs.get_data() * 1000  # shape is (n_trials, n_channels, n_samples)
    Y = epochs.events[:, -1]
    del epochs
    # split the dataset into train/validate/test
    lfist_idx = np.where(Y == EVENT_ID["lfist"])[0]
    rfist_idx = np.where(Y == EVENT_ID["rfist"])[0]
    assert lfist_idx.size == rfist_idx.size
    size = lfist_idx.size
    # shuffle events to avoid selecting all the events at the beginning or end of the
    # dataset
    rng = np.random.default_rng()
    rng.shuffle(lfist_idx)
    rng.shuffle(rfist_idx)
    # use 70% events for training, 15% for validation and 15% for test.
    # shuffle the indices again to shuffle the left/right events from the horizontal
    # stack.
    n2 = int(0.7 * size)
    idx_train = np.hstack((lfist_idx[:n2], rfist_idx[:n2]))
    rng.shuffle(idx_train)
    X_train = X[idx_train, :, :]
    Y_train = Y[idx_train]
    n1 = n2
    n2 = n1 + int(0.15 * size)
    idx_val = np.hstack((lfist_idx[n1:n2], rfist_idx[n1:n2]))
    rng.shuffle(idx_val)
    X_validate = X[idx_val, :, :]
    Y_validate = Y[idx_val]
    n1 = n2
    idx_test = np.hstack((lfist_idx[n1:], rfist_idx[n1:]))
    rng.shuffle(idx_test)
    X_test = X[idx_test, :, :]
    Y_test = Y[idx_test]
    # sanity-checks
    assert idx_train.size + idx_val.size + idx_test.size == 2 * size
    assert np.unique(np.hstack((idx_train, idx_val, idx_test))).size == 2 * size
    return X_train, Y_train, X_validate, Y_validate, X_test, Y_test


def _fit_EEGNet(
    model: Model | str | Path | None,
    X_train: NDArray[np.float64],
    Y_train: NDArray[np.int64],
    X_validate: NDArray[np.float64],
    Y_validate: NDArray[np.int64],
    X_test: NDArray[np.float64],
    Y_test: NDArray[np.int64],
) -> Model:
    """Fit EEGNet model with the dataset recorded with offline_calibration.

    Parameters
    ----------
    model : Model | path-like | None
        If provided, model on which fit is resumed. If None, a new model is created.
    X_train : array of shape (n_trials, n_channels, n_samples)
        Training dataset.
    Y_train : array of shape (n_trials,)
        Training labels.
    X_validate : array of shape (n_trials, n_channels, n_samples)
        Validation dataset.
    Y_validate : array of shape (n_trials,)
        Validation labels.
    X_test : array of shape (n_trials, n_channels, n_samples)
        Test dataset.
    Y_test : array of shape (n_trials,)
        Test labels.

    Returns
    -------
    model : Model
        Fitted EEGNet model.
    """
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.utils import to_categorical

    from ._bci_EEGNet import EEGNet

    check_type(model, (Model, str, Path, None), "model")
    if isinstance(model, (str, Path)):
        model = ensure_path(model, must_exist=True)
        model = load_model(model)

    # convert labels to one-hot encodings.
    Y_train = to_categorical(Y_train - 1)
    Y_validate = to_categorical(Y_validate - 1)
    Y_test = to_categorical(Y_test - 1)

    # convert data to NHWC (n_trials, n_channels, n_samples, n_kernels) format.
    assert X_train.shape[1] == X_validate.shape[1] == X_test.shape[1]
    assert X_train.shape[2] == X_validate.shape[2] == X_test.shape[2]
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]
    n_kernels = 1
    X_train = X_train.reshape(X_train.shape[0], n_channels, n_samples, n_kernels)
    X_validate = X_validate.reshape(
        X_validate.shape[0], n_channels, n_samples, n_kernels
    )
    X_test = X_test.reshape(X_test.shape[0], n_channels, n_samples, n_kernels)

    # create and fit model
    if model is None:
        model = EEGNet(
            len(EVENT_ID),
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
    fname_chkpoint = str(
        Path(tempdir.name) / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime())}.h5"
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
    del tempdir

    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    logger.info("Classification accuracy [test]: %f", acc)
    return model


def fit_EEGNet(fname: str | Path) -> Model:
    """Fit EEGNet model with the dataset recorded in the calibration.

    Parameters
    ----------
    fname : path-like
        Path to the FIFF recording.

    Returns
    -------
    model : Model
        Fitted EEGNet model.
    """
    epochs = _load_dataset(fname)
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = _get_data(epochs)
    model = _fit_EEGNet(None, X_train, Y_train, X_validate, Y_validate, X_test, Y_test)
    return model
