import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.skip import SkipLayer
from hourglass_tensorflow.layers.conv_block import ConvBlockLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer


class ResidualLayer(Layer):
    def __init__(
        self,
        output_filters: int,
        momentum: float = 0.97,
        epsilon: float = 0.001,
        name: str = None,
        #dtype=None,
        #dynamic=False,
        trainable: bool = True,
        use_last_relu: bool = False,
    ) -> None:
        #super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        # Batch Norm layer 
        #self.batch_norm = layers.BatchNormalization(
        #    axis=-1,
        #    momentum=momentum,
        #    epsilon=epsilon,
        #    trainable=trainable,
        #    name="BatchNorm",
        #)

        # Conv Layer
        #self.conv_layer = layers.Conv2D(
        #    filters=self.output_filters,
        #    kernel_size=1,
        #    strides=1,
        #    padding="same",
        #    name="Conv2D",
        #    activation=None,
        #    kernel_initializer="glorot_uniform",
        #)

        # Convolutional block
        self.conv_block = ConvBlockLayer(
            output_filters=output_filters,
            momentum=momentum,
            epsilon=epsilon,
            name="ConvBlock",
            #dtype=dtype,
            #dynamic=dynamic,
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
        self.relu= layers.ReLU(name="ReLU",)  if self.use_last_relu else lambda x:x
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
        #_inputs = self.batch_norm(inputs,training=training)
        #_inputs = self.conv_layer(_inputs ,training=training)
        _sum = self.add(
            [
                self.conv_block(inputs, training=training),
                #self.skip(inputs, training=training),
                inputs,
            ])
        return self.relu(_sum)
    def build(self, input_shape):
        self.built = True
