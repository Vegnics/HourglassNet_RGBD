import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.saving import register_keras_serializable

@register_keras_serializable(package="lSkip")
class SkipLayer(Layer):
    def __init__(
        self,
        output_filters: int,
        name: str = None,
        trainable: bool = True,
        **kwargs,
        #momentum: float = 0.98,
        #epsilon: float = 1e-3,
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.output_filters = output_filters
        # Create Layers
        #self.conv = None
        self.conv = layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name="Conv2D",
            activation=None,
            kernel_initializer="glorot_uniform",
        )
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
            },
        }

    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        return self.conv(inputs)

    def build(self, input_shape):
        super().build(input_shape)
    
    """@classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """