"""Define architecture and input constants"""


class Constants:

    # MLP Constants
    _MLP_NUM_FEATURES = 67
    _MLP_NUM_LAYERS = 3
    _MLP_NUM_DIMS = 32

    # TCN Constants
    _TCN_LENGTH = 42
    _TCN_NUM_CHANNELS = 11
    _TCN_NUM_FILTERS = 128
    _TCN_KERNEL_SIZE = 3
    _TCN_NUM_STACK = 2
    _TCN_DIALATIONS = [1, 2, 4]
    _TCN_PADDING = 'same'


    _NUM_TARGETS = 24
