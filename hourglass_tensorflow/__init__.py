__version__ = "1.0.1"

from hourglass_tensorflow import *

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "Please install tensorflow before using this package: pip install tensorflow"
    )
