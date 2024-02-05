import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.skip import SkipLayer
from hourglass_tensorflow.layers.conv_block import ConvBlockLayer


class ResidualLayer(Layer):
    def __init__(
        self,
        output_filters: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
        use_last_relu: bool = True,
    ) -> None:
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        self.layers = []
        # Input Conv Layer
        self.conv_layer = layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name="Conv2D",
            activation="relu",
            kernel_initializer="glorot_uniform",
        )

        # Create Convolutional block
        self.conv_block = ConvBlockLayer(
            output_filters=output_filters,
            momentum=momentum,
            epsilon=epsilon,
            name="ConvBlock",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
        )
        #self.skip = SkipLayer(
        #    output_filters=output_filters,
        #    name="Skip",
        #    dtype=dtype,
        #    dynamic=dynamic,
        #    trainable=trainable,
        #)
        self.add = layers.Add(name="Add")
        self.relu = layers.ReLU(
            name="ReLU",
        )

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        _inputs = self.conv_layer(inputs ,training=training)
        _sum = self.add(
            [
                self.conv_block(_inputs, training=training),
                #self.skip(inputs, training=training),
                _inputs,
            ])
        return self.relu(_sum) if self.use_last_relu else _sum
    def build(self, input_shape):
        pass
