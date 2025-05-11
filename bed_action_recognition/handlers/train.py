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

from hourglass_tensorflow.callbacks.dummycallback import DummyCallback

# region Abstract Class

R = TypeVar("R")

class BedARTF_TrainHandler():
    def __init__(self,epochs,batch_size):
        self._epochs = epochs
        #self._epoch_size = self.config.epoch_size
        self._batch_size = batch_size
        self._metrics = [KERAS.metrics.CategoricalAccuracy()]
        self._callbacks = [KERAS.callbacks.ModelCheckpoint("bed_action_recognition/data/model_bedar.keras",monitor="val_categorical_accuracy",save_best_only=True,mode="max",initial_value_threshold=0.0,verbose=1)]
        self._loss = KERAS.losses.CategoricalCrossentropy()
        self._learning_rate = KERAS.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.5e-3,
                                                                          decay_steps=300,
                                                                          decay_rate=0.98)
        self._optimizer = KERAS.optimizers.Adam(learning_rate=self._learning_rate,
                                                beta_1=0.9,
                                                beta_2=0.99,
                                                epsilon=1e-6)
    """
    def _instantiate(self, obj: HTFObjectReference[R], **kwargs) -> R:
        if isinstance(obj, HTFObjectReference):
            return obj.init(**kwargs)
        else:
            return obj
    """

    def compile(self, model: Model, *args, **kwargs) -> None:
        print("Compiling the models ...")
        model.compile(optimizer=self._optimizer, metrics=self._metrics, loss=self._loss, jit_compile=False)
        #dummy.compile(optimizer=self._optimizer, metrics=self._metrics, loss=self._loss, jit_compile=False)

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
        #dummyds = tf.data.Dataset.from_tensor_slices(tf.random.normal((40,7,256,256,4)))
        #dummyds = dummyds.batch(15)
        #self._callbacks.append(DummyCallback(dummyds))
        with tf.device('/GPU:0'):
            tds_card = 1000
            train_dataset = train_dataset.shuffle(tds_card,reshuffle_each_iteration=True)
            train_dataset = train_dataset.repeat(6)
            batch_train = self._apply_batch(train_dataset) 
            #print("   ??????    >>>BATCH TRAIN: ",batch_train)
            batch_validation = validation_dataset.batch(80)#self._apply_batch(validation_dataset)
            batch_num = batch_train.__len__()
            print("BATCH INFO :", batch_num.numpy().tolist(),(batch_num//self._epochs).numpy().tolist())
            model(tf.random.normal((1,7,256,256,4))) 
            model.summary()
            model.fit(
                batch_train,
                epochs=self._epochs,
                #steps_per_epoch=self._epoch_size,
                steps_per_epoch=int(batch_num//self._epochs),
                shuffle=True,
                validation_data=batch_validation,
                #validation_steps=100,
                callbacks=self._callbacks,
            )
           

