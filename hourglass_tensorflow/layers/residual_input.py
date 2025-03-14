import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.skip import SkipLayer
from hourglass_tensorflow.layers.conv_block import ConvBlockLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer
from hourglass_tensorflow.layers.batch_norm_conv_1 import BatchNormConv1Layer

class ResidualLayerIn(Layer):
    def __init__(
        self,
        output_filters: int,
        momentum: float = 0.98,
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
        # Input layer (used just to match the dimensionality)
        self.match_layer =  None
        # Convolutional block
        self.conv_block = None

        self.add = None
        self.relu= None
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_last_relu": self.use_last_relu 
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        _inputs = self.match_layer(inputs,training=training)
        _sum = self.add(
            [
                self.conv_block(inputs, training=training),
                #self.skip(inputs, training=training),
                _inputs,#skip,
            ])
        return self.relu(_sum)
    def build(self, input_shape):
        self.match_layer =   SkipLayer(
            output_filters=self.output_filters,
            name="Skip",
            #dtype=dtype,
            #dynamic=dynamic,
            trainable=self.trainable,
        )

        # Convolutional block
        self.conv_block = ConvBlockLayer(
            output_filters=self.output_filters,
            momentum=self.momentum,
            epsilon=self.epsilon,
            name="ConvBlock",
            #dtype=dtype,
            #dynamic=dynamic,
            trainable=self.trainable,
        )

        self.add = layers.Add(name="Add")
        self.relu= layers.ReLU(name="ReLU",)  if self.use_last_relu else lambda x:x
        super().build(input_shape) 
        self.built = True
    @classmethod
    def from_config(cls, config):
        return cls(**config)
