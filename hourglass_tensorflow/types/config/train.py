from typing import List
from typing import Union
from typing import Optional

import keras as KERAS
if KERAS.__version__ < "2.18.0":
    from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule
else:
    from keras.optimizers.schedules import LearningRateSchedule
    
from pydantic import Field
from keras.losses import Loss
from keras.metrics import Metric
from keras.callbacks import Callback
from keras.optimizers import Optimizer

from hourglass_tensorflow.types.config.fields import HTFConfigField
from hourglass_tensorflow.types.config.fields import HTFObjectReference


class HTFTrainConfig(HTFConfigField):
    epochs: int = 10
    epoch_size: int = 1000
    batch_size: Optional[int] = None
    learning_rate: Union[HTFObjectReference[LearningRateSchedule], float] = 0.00025
    loss: Union[HTFObjectReference[Loss], str] = "binary_crossentropy"
    optimizer: Union[HTFObjectReference[Optimizer], str] = "rmsprop"
    callbacks: List[HTFObjectReference[Callback]] = Field(default=list)
    metrics: List[HTFObjectReference[Metric]] = Field(default=list)
    object: Optional[HTFObjectReference] = Field(
        default=HTFObjectReference(
            source="hourglass_tensorflow.handlers.train.HTFTrainHandler"
        )
    )
    use_2jointHM: bool = False
    class Config:
        arbitrary_types_allowed = True
