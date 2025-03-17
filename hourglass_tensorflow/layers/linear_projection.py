import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="lLinearProj")
class LinearProjection(Layer):
    """
    This layer performs 2D convolution, Batch Normalization, and ReLU.
    """
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        activation: str = None,
        kernel_initializer: str = "glorot_uniform",
        outmax: float = None,
        name: str = None,
        trainable: bool = True,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.outmax = outmax
        # Create layers
        self.conv = None
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
                "outmax": self.outmax
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        # 我的Method
        #x = self.batch_norm(inputs,training=training)
        #x = self.conv(x)
        #return self.relu(x)
        #
        x = self.conv(inputs)     
        return x

    def build(self, input_shape):
        print(f"[DEBUG]: {self.name} -- input shape : {input_shape}")
        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            name="Conv2D",
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
        )
        super().build(input_shape) 
        self.built = True
    @classmethod
    def from_config(cls, config):
        instance =  cls(**config)
        instance.conv = None
        return instance
