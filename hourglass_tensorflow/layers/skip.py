import tensorflow as tf
from keras import layers
from keras.layers import Layer


class SkipLayer(Layer):
    def __init__(
        self,
        output_filters: int,
        name: str = None,
        #dtype=None,
        #dynamic=False, 
        trainable: bool = True,
        momentum: float = 0.98,
        epsilon: float = 1e-3,
    ) -> None:
        #super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.output_filters = output_filters
        # Create Layers
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

        #self.relu =layers.ReLU(
        #    name="ReLU",
        #) #<- lambda x:x

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        #if inputs.get_shape()[-1] == self.output_filters:
        #    return inputs
        #else:
        #    v = self.conv(inputs)
        #    return v
        
        #x = self.batch_norm(inputs)
        return self.conv(inputs)

    def build(self, input_shape):
        self.built = True
