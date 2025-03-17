import tensorflow as tf
from keras import layers
from keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="lSkip")
class SkipLayer(Layer):
    def __init__(
        self,
        output_filters: int,
        name: str = None,
        trainable: bool = True,
        #momentum: float = 0.98,
        #epsilon: float = 1e-3,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.output_filters = output_filters
        # Create Layers
        self.conv = None
        #self.batch_norm = layers.BatchNormalization(
        #    axis=-1,
        #    momentum=momentum,
        #    epsilon=epsilon,
        #    trainable=trainable,
        #    name="BatchNorm",
        #)

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "trainable": self.trainable
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        return self.conv(inputs)

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name="Conv2D",
            activation=None,
            kernel_initializer="glorot_uniform",
        )
        self.built = True
    
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        instance.conv =  None
        return instance
