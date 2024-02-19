from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np
from mne import Epochs, concatenate_epochs, find_events
from mne.io import read_raw_fif

from ..utils._checks import ensure_path

if TYPE_CHECKING:
    from pathlib import Path

    from mne import BaseEpochs
    from numpy.typing import NDArray


_EVENT_ID: dict[str, int] = dict(lfist=1, rfist=2, hands_open=3)


def _load_dataset(fname: str | Path) -> BaseEpochs:
    """Load and preprocess the EEG dataset from the DSI-24 amplifier.

    Parameters
    ----------
    fname : path-like
        Path to the FIFF recording.

    Returns
    -------
    epochs : Epochs
        Epochs filtered and resampled for the 3 events, 'lfist', 'rfist' and
        'hands_open'.
    """
    fname = ensure_path(fname, must_exist=True)
    raw = read_raw_fif(fname, preload=False)
    raw.drop_channels(["X1", "X2", "X3", "A2"])
    raw.load_data()
    raw.set_montage("standard_1020")
    raw.filter(
        l_freq=2.0,
        h_freq=25.0,
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge",
    )
    # create epochs and resample to 128 Hz
    events = find_events(raw, stim_channel="TRIGGER")
    epochs_list = list()
    for step in np.arange(0, 1.1, 0.1):
        epochs = Epochs(
            raw,
            events,
            _EVENT_ID,
            tmin=0.5 + step,
            tmax=1.5 + step,
            baseline=None,
            preload=True,
        )
        epochs.shift_time(tshift=0, relative=False)
        epochs_list.append(epochs)
    epochs = concatenate_epochs(epochs_list)
    epochs.resample(128)
    return epochs


def _get_data_and_labels(
    epochs: BaseEpochs,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Extract labels and dataset from the epochs."""
    # the raw data is scaled by 1000 due to scaling sensitivity in deep learning
    labels = epochs.events[:, -1]
    X = epochs.get_data() * 1000  # shape is (n_trials, n_channels, n_samples)
    Y = labels
    # split the dataset into train/validate/test
    lfist_idx = np.where(Y == _EVENT_ID["lfist"])[0]
    rfist_idx = np.where(Y == _EVENT_ID["rfist"])[0]
    hands_open_idx = np.where(Y == _EVENT_ID["hands_open"])[0]
    assert lfist_idx.size == rfist_idx.size == hands_open_idx.size
    size = lfist_idx.size
    # shufle events to avoid selecting all the events at the beginning or end of the
    # dataset
    rng = np.random.default_rng()
    rng.shuffle(lfist_idx)
    rng.shuffle(rfist_idx)
    rng.shuffle(hands_open_idx)
    # use 70% events for training, 15% for validation and 15% for test.
    # shuffle the indices again to shuffle the left/right/hands-open events from the
    # horizontal stack.
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
    # sanity-checks
    assert idx_train.size + idx_val.size + idx_test.size == 3 * size
    assert np.unique(np.hstack((idx_train, idx_val, idx_test))).size == 3 * size
    return X_train, Y_train, X_validate, Y_validate, X_test, Y_test
