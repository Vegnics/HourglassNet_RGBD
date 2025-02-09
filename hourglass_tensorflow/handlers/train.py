from abc import abstractmethod
from typing import List
from typing import Union
from typing import TypeVar

import tensorflow as tf
import numpy as np
from keras.losses import Loss
from keras.models import Model
from keras.metrics import Metric
from keras.callbacks import Callback
from keras.optimizers import Optimizer
import keras as KERAS
if KERAS.__version__ < "2.18.0":
    from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule
else:
    from keras.optimizers.schedules import LearningRateSchedule

from hourglass_tensorflow.types.config import HTFTrainConfig
from hourglass_tensorflow.types.config import HTFObjectReference
from hourglass_tensorflow.handlers.meta import _HTFHandler

# region Abstract Class

R = TypeVar("R")


class _HTFTrainHandler(_HTFHandler):
    def __init__(
        self,
        config: HTFTrainConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config=config, *args, **kwargs)
        self._epochs: int = None
        self._epoch_size: int = None
        self._batch_size: int = None
        self._learning_rate: Union[
            HTFObjectReference[LearningRateSchedule], float
        ] = None
        self._loss: Union[HTFObjectReference[Loss], str] = None
        self._optimizer: Union[HTFObjectReference[Optimizer], str] = None
        self._metrics: List[HTFObjectReference[Metric]] = None
        self._callbacks: List[HTFObjectReference[Callback]] = None
        self.init_handler()

    @property
    def config(self) -> HTFTrainConfig:
        return self._config

    @abstractmethod
    def compile(self, model: Model, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        model: Model,
        train_dataset: tf.data.Dataset = None,
        test_dataset: tf.data.Dataset = None,
        validation_dataset: tf.data.Dataset = None,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        self.compile(*args, **kwargs)
        self.fit(*args, **kwargs)


# enregion

# region Handler


class HTFTrainHandler(_HTFTrainHandler):
    def _instantiate(self, obj: HTFObjectReference[R], **kwargs) -> R:
        if isinstance(obj, HTFObjectReference):
            return obj.init(**kwargs)
        else:
            return obj

    def init_handler(self, *args, **kwargs) -> None:
        self._epochs = self.config.epochs
        self._epoch_size = self.config.epoch_size
        self._batch_size = self.config.batch_size
        self._learning_rate = self._instantiate(self.config.learning_rate)
        self._loss = self._instantiate(self.config.loss)
        self._optimizer = self._instantiate(
            self.config.optimizer, learning_rate=np.float32(self._learning_rate)
        )
        self._metrics = [obj.init() for obj in self.config.metrics]
        self._callbacks = [obj.init() for obj in self.config.callbacks]

    def compile(self, model: Model, *args, **kwargs) -> None:
        model.compile(optimizer=self._optimizer, metrics=self._metrics, loss=self._loss, jit_compile=False)
    def _apply_batch(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        if isinstance(dataset, tf.data.Dataset):
            return dataset.batch(self._batch_size)
    def fit(
        self,
        model: Model,
        train_dataset: tf.data.Dataset = None,
        test_dataset: tf.data.Dataset = None,
        validation_dataset: tf.data.Dataset = None,
        *args,
        **kwargs,
    ) -> None:
        with tf.device('/GPU:0'):
            _ = self._apply_batch(test_dataset)

            #tds_card = int(train_dataset.cardinality().numpy())
            #vds_card = int(validation_dataset.cardinality().numpy())
            #print("CARDINALITY",tds_card,vds_card)
            tds_card = 2400
            vds_card = 300
            train_dataset = train_dataset.shuffle(tds_card,reshuffle_each_iteration=True)
            validation_dataset = validation_dataset.shuffle(vds_card,reshuffle_each_iteration=False)
            train_dataset = train_dataset.repeat(7) #7
            #validation_dataset = validation_dataset.repeat(2)
            batch_train = self._apply_batch(train_dataset) 
            #print("   ??????    >>>BATCH TRAIN: ",batch_train)
            batch_validation = validation_dataset.batch(150)#self._apply_batch(validation_dataset)
            batch_num = batch_train.__len__()
            print("BATCH INFO :", batch_num.numpy().tolist(),(batch_num//self._epochs).numpy().tolist())
            model.summary()
            model.fit(
                batch_train,
                epochs=self._epochs,
                #steps_per_epoch=self._epoch_size,
                steps_per_epoch=int(batch_num//self._epochs),
                shuffle=True,
                validation_data=batch_validation,
                callbacks=self._callbacks,
            )


# endregion
