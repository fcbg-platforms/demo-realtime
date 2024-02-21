from pathlib import Path

import numpy as np
from mne import BaseEpochs
from numpy.typing import NDArray
from tensorflow.keras.models import Model

from ..utils._checks import check_type as check_type
from ..utils._checks import ensure_path as ensure_path
from ..utils.logs import logger as logger
from ._config import EVENT_ID as EVENT_ID

def _load_dataset(fname: str | Path) -> BaseEpochs:
    """Load and preprocess the EEG dataset from the DSI-24 amplifier.

    Parameters
    ----------
    fname : path-like
        Path to the FIFF recording.

    Returns
    -------
    epochs : Epochs
        Epochs preprocessed for the 3 events, 'lfist', 'rfist' and 'hands_open'.
    """

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
        Epochs preprocessed for the 3 events, 'lfist', 'rfist' and 'hands_open'.

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
