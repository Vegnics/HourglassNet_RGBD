import math

import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.residual import ResidualLayer
from hourglass_tensorflow.layers.residual_input import ResidualLayerIn
#from hourglass_tensorflow.layers.residual_2 import ResidualLayerSkip as ResidualLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer
from hourglass_tensorflow.layers.batch_norm_conv_relu import BatchNormConvReluLayer


class DownSamplingLayer(Layer):
    """
    This is the downsampling layer. The one which receives the input image with a size of 
    256x256 and transform it into 256 features of size 64x64.
    """
    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 64,
        kernel_size: int = 7,
        output_filters: int = 256,
        name: str = None,
        #dtype=None,
        #dynamic=False,
        trainable: bool = True,
    ) -> None:
        #super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.output_filters = output_filters
        # Init Computation
        self.downsamplings = int(math.log2(input_size // output_size) + 1)
        self.layers = []
        # Create Layers
        for i in range(self.downsamplings):
            if i == 0:
                self.layers.append(
                    ConvBatchNormReluLayer(
                    #BatchNormConvReluLayer(
                        filters=(
                            output_filters // 4
                            if self.downsamplings > 1
                            else output_filters
                        ),
                        kernel_size=kernel_size,
                        strides=(2 if self.downsamplings > 1 else 1),
                        name="CNBR",
                        #dtype=dtype,
                        #dynamic=dynamic,
                        trainable=trainable,
                        use_relu=True,
                    )
                )
            elif i == self.downsamplings - 1:
                self.layers.append(
                    ResidualLayer(
                        output_filters=output_filters // 2,
                        name=f"Residual{i}",
                        #dtype=dtype,
                        #dynamic=dynamic,
                        trainable=trainable,
                    )
                )
                self.layers.append(
                    ResidualLayerIn(
                        output_filters=output_filters,
                        name=f"Residual{i}",
                        #dtype=dtype,
                        #dynamic=dynamic,
                        trainable=trainable,
                    )
                )
            else:
                self.layers.append(
                    ResidualLayerIn(
                        output_filters=output_filters // 2,
                        name=f"Residual{i}",
                        #dtype=dtype,
                        #dynamic=dynamic,
                        trainable=trainable,
                    )
                )
                self.layers.append(
                    layers.MaxPool2D(
                        pool_size=(2, 2), padding="valid", name=f"MaxPool{i}"
                    )
                )

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "input_size": self.input_size,
                "output_size": self.output_size,
                "kernel_size": self.kernel_size,
                "output_filters": self.output_filters,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = tf.cast(inputs,dtype=tf.dtypes.float32)
        for layer in self.layers:
            x = layer(x, training=training)
        return x
    def build(self, input_shape):
        self.built = True
