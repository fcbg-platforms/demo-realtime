from pathlib import Path

import numpy as np
from mne.io import read_raw_fif
from mne import Epochs, concatenate_epochs, find_events
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from demo_realtime import logger, set_log_level
from demo_realtime.bci_motor_decoding import offline_calibration, online

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore


set_log_level("INFO")

directory = Path.home() / "Downloads" / "bci"
fname = offline_calibration(10, "WS-default", directory)

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

# extract raw data and labels. The raw data is scaled by 1000 due to
# scaling sensitivity in deep learning.
labels = epochs.events[:, -1]
X = epochs.get_data() * 1000  # shape is (n_trials, n_channels, n_samples)
Y = labels

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
idx_train = np.hstack((lfist_idx[:n2], rfist_idx[:n2], hands_open_idx[:n2]))
rng.shuffle(idx_train)
X_train = X[idx_train, :, :]
Y_train = Y[idx_train]

n1 = n2
n2 = n1 + int(0.15 * size)
idx_val = np.hstack(
    (lfist_idx[n1:n2], rfist_idx[n1:n2], hands_open_idx[n1:n2])
)
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
Y_train = np_utils.to_categorical(Y_train - 1)
Y_validate = np_utils.to_categorical(Y_validate - 1)
Y_test = np_utils.to_categorical(Y_test - 1)

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

model = (
    files("demo_realtime")
    / "resources"
    / "bci_motor_decoding-model-mscheltienne"
)
assert model.exists()  # sanity-check
model = load_model(model)
checkpointer = ModelCheckpoint(
    filepath=directory / "checkpoint-demo.h5",
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
model.load_weights(directory / "checkpoint-demo.h5")

probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
logger.info("Classification accuracy [test]: %f", acc)

online("WS-default", model, duration=300)
