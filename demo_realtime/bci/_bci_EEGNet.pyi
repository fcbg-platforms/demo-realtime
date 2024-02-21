from ..utils._checks import check_type as check_type
from ..utils._checks import check_value as check_value
from ..utils._imports import import_optional_dependency as import_optional_dependency

def EEGNet(
    n_classes: int,
    n_channels: int,
    n_samples: int,
    dropoutRate: float = 0.5,
    kernelLength: int = 64,
    F1: int = 8,
    D: int = 2,
    F2: int = 16,
    norm_rate: float = 0.25,
    dropoutType: str = "Dropout",
):
    """Keras Implementation of EEGNet.

    Details can be found in the associated paper:
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.

    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128 Hz. If you want to use this
    model for any other sampling rate you will need to modify the lengths of
    temporal kernels and average pooling size in blocks 1 and 2 as needed
    (double the kernel lengths for double the sampling rate, etc). Note that we
    haven't tested the model performance with this rule so this may not work
    well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
        advised to do some model searching to get optimal performance on your
        particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of
    this parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D
    for overcomplete). We believe the main parameters to focus on are F1 and D.

    Parameters
    ----------
    n_classes : int
        Number of classes to classify.
    n_channels : int
        Number of channels in the EEG dataset.
    n_samples : int
        Number of samples in the EEG dataset. Since the input signal should be
        sampled at 128 Hz, a 1 second epoch yields 128 samples.
    dropoutRate : float
        The dropout fraction between 0 and 1.
    kernelLength : int
        Length of the temporal convolution in the first layer. We found that
        setting this to be half the sampling rate worked well in practice. For
        the SMR dataset in particular since the data was high-passed at 4 Hz we
        used a kernel length of 32.
    F1 : int
        Number of temporal filters to learn.
    D : int
        Number of spatial filters to learn within each temporal convolution.
    F2 : int
        Number of pointwise filters to learn. Default ``F1 * D``.
    norm_rate : float
        Maximum norm value for the incoming weights used to constraint the last
        dense layer.
    dropoutType : ``SpatialDropout2D`` | ``Dropout``
        Type of dropout to use.

    Returns
    -------
    model : Model
        The Keras model.
    """
