import pytest

from demo_realtime.bci._bci_EEGNet import EEGNet


def test_EEGNet():
    """Test the creation of the EEGNet model."""
    pytest.importorskip("tensorflow")

    from tensorflow.keras.models import Model

    model = EEGNet(n_classes=4, n_channels=64, n_samples=128, dropoutType="Dropout")
    assert isinstance(model, Model)
    model = EEGNet(
        n_classes=4,
        n_channels=64,
        n_samples=128,
        dropoutType="SpatialDropout2D",
    )
    assert isinstance(model, Model)


def test_invalid_arguments():
    """Test the creation of the EEGNet model with invalid arguments."""
    pytest.importorskip("tensorflow")

    from tensorflow.keras.layers import Dropout

    with pytest.raises(TypeError, match="'n_classes' must be"):
        EEGNet(n_classes=4.0, n_channels=64, n_samples=128, dropoutType="Dropout")
    with pytest.raises(AssertionError):
        EEGNet(
            n_classes=-4,
            n_channels=64,
            n_samples=128,
        )

    with pytest.raises(TypeError, match="'n_channels' must be"):
        EEGNet(n_classes=4, n_channels=64.0, n_samples=128, dropoutType="Dropout")
    with pytest.raises(AssertionError):
        EEGNet(
            n_classes=4,
            n_channels=-64,
            n_samples=128,
        )

    with pytest.raises(TypeError, match="'n_samples' must be"):
        EEGNet(n_classes=4, n_channels=64, n_samples=128.0, dropoutType="Dropout")
    with pytest.raises(AssertionError):
        EEGNet(
            n_classes=4,
            n_channels=64,
            n_samples=-128,
        )

    with pytest.raises(TypeError, match="'dropoutRate' must be"):
        EEGNet(
            n_classes=4,
            n_channels=64,
            n_samples=128,
            dropoutRate="0.5",
        )
    with pytest.raises(AssertionError):
        EEGNet(
            n_classes=4,
            n_channels=64,
            n_samples=128,
            dropoutRate=1.1,
        )
    with pytest.raises(AssertionError):
        EEGNet(
            n_classes=4,
            n_channels=64,
            n_samples=128,
            dropoutRate=-0.1,
        )

    with pytest.raises(TypeError, match="'kernelLength' must be"):
        EEGNet(n_classes=4, n_channels=64, n_samples=128, kernelLength=32.0)
    with pytest.raises(AssertionError):
        EEGNet(n_classes=4, n_channels=64, n_samples=128, kernelLength=-32)

    with pytest.raises(TypeError, match="'F1' must be"):
        EEGNet(n_classes=4, n_channels=64, n_samples=128, F1=8.0)
    with pytest.raises(AssertionError):
        EEGNet(n_classes=4, n_channels=64, n_samples=128, F1=-8)

    with pytest.raises(TypeError, match="'D' must be"):
        EEGNet(n_classes=4, n_channels=64, n_samples=128, D=2.0)
    with pytest.raises(AssertionError):
        EEGNet(n_classes=4, n_channels=64, n_samples=128, D=-2)

    with pytest.raises(TypeError, match="'F2' must be"):
        EEGNet(n_classes=4, n_channels=64, n_samples=128, F2=16.0)
    with pytest.raises(AssertionError):
        EEGNet(n_classes=4, n_channels=64, n_samples=128, F2=-16)

    with pytest.raises(TypeError, match="'dropoutType' must be"):
        EEGNet(n_classes=4, n_channels=64, n_samples=128, dropoutType=Dropout)
    with pytest.raises(ValueError, match="value for the 'dropoutType'"):
        EEGNet(n_classes=4, n_channels=64, n_samples=128, dropoutType="101")
