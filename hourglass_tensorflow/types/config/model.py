from typing import List
from typing import Union
from typing import Literal
from typing import Optional
from typing import TypedDict

import keras.layers
import keras.models
from pydantic import Field

from hourglass_tensorflow.types.config.fields import HTFConfigField
from hourglass_tensorflow.types.config.fields import HTFObjectReference

DATA_FORMAT = Union[
    Literal["NHWC"],
    Literal["NCHW"],
]

ATTENTION_MECHANISMS = Union[
    Literal["FAM"],
    Literal["SAM"],
    Literal["NoAM"],
]

class HTFModelHandlerReturnObject(TypedDict):
    inputs: keras.layers.Layer
    outputs: keras.layers.Layer
    model: keras.models.Model


class HTFModelAsLayers(TypedDict):
    downsampling: keras.layers.Layer
    hourglasses: List[keras.layers.Layer]
    outputs: keras.layers.Layer
    model: keras.models.Model


class HTFModelParams(HTFConfigField):
    name: str = "HourglassNetwork"
    input_size: int = 256
    output_size: int = 64
    channel_number: int = 3
    stages: int = 4
    stage_filters: int = 128
    channels_1joint: int = 16
    channels_2joint: int = 16
    downsamplings_per_stage: int = 4
    intermediate_supervision: bool = True
    skip_AM: ATTENTION_MECHANISMS = "NoAM"
    s2f_AM: ATTENTION_MECHANISMS = "NoAM"
    f2s_AM: ATTENTION_MECHANISMS = "NoAM"
    use_2jointHM: bool = False
    use_kernel_regularization: bool = False
    freeze_attention_weights: bool = False

class HTFModelConfig(HTFConfigField):
    object: Optional[HTFObjectReference] = Field(
        default=HTFObjectReference(
            source="hourglass_tensorflow.handlers.model.HTFModelHandler"
        )
    )
    load_model: bool = False
    model_path: str = ""
    build_as_model: bool = False
    data_format: DATA_FORMAT = "NHWC"
    params: Optional[HTFModelParams] = Field(default=HTFModelParams)
    batch_size: int = 1
