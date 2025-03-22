from logging import debug
import math

import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.saving import register_keras_serializable

from hourglass_tensorflow.layers.residual import ResidualLayer,ResidualLayerIn
#from hourglass_tensorflow.layers.residual_2 import ResidualLayerSkip as ResidualLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer

@register_keras_serializable(package="lDownsampling")
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
        residual_nblocks: int = None,
        trainable: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.output_filters = output_filters
        self.residual_nblocks = residual_nblocks
        
        # Init Computation
        self.downsamplings = int(math.log2(input_size // output_size) + 1)
        self.layer_list = []

        # Create Layers
        for i in range(self.downsamplings):
            if i == 0:
                self.layer_list.append(
                    ConvBatchNormReluLayer(
                    #BatchNormConvReluLayer(
                        filters=(
                            self.output_filters // 4
                            if self.downsamplings > 1
                            else self.output_filters
                        ),
                        kernel_size=self.kernel_size,
                        strides=(2 if self.downsamplings > 1 else 1),
                        name="CNBR",
                        padding="same",
                        trainable=trainable,
                        use_relu=True,
                    )
                )
            elif i == self.downsamplings - 1:
                self.layer_list.append(
                    ResidualLayer(
                        output_filters=self.output_filters // 2,
                        nblocks=self.residual_nblocks,
                        name=f"Residual{i}_last",
                        trainable=trainable,
                        attentionType="NoAM",
                    )
                )
                self.layer_list.append(
                    ResidualLayerIn(
                        output_filters=self.output_filters,
                        nblocks=self.residual_nblocks,
                        name=f"Residual{i}_in",
                        trainable=trainable,
                    )
                )
            else:
                self.layer_list.append(
                    ResidualLayerIn(
                        output_filters=self.output_filters // 2,
                        nblocks=self.residual_nblocks,
                        name=f"Residual{i}_middle",
                        trainable=trainable,
                    )
                )
                self.layer_list.append(
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
                "residual_nblocks": self.residual_nblocks,
            },
        }

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        #x = tf.cast(inputs,dtype=tf.dtypes.float32)
        x = 1.0*inputs
        for layer in self.layer_list:
            x = layer(x, training=training)
        return x
    
    def build(self, input_shape):
        #print(f"[DEBUG]: {self.name} -- input shape : {input_shape}")
        super().build(input_shape)
        self.built = True

    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """