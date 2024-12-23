from typing import List
from typing import Union
from typing import Optional

from pydantic import Field
from keras.losses import Loss
from keras.metrics import Metric
from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule
#from keras.optimizers.schedules import LearningRateSchedule

from hourglass_tensorflow.types.config.fields import HTFConfigField
from hourglass_tensorflow.types.config.fields import HTFObjectReference


class HTFTestConfig(HTFConfigField):
    batch_size: Optional[int] = None
    object: Optional[HTFObjectReference] = Field(
        default=HTFObjectReference(
            source="hourglass_tensorflow.handlers.test.HTFTestHandler"
        )
    )
    class Config:
        arbitrary_types_allowed = True
