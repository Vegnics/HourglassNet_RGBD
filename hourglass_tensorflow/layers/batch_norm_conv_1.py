import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish


class BatchNormConv1Layer(Layer):
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
        kernel_initializer: str = "glorot_normal",
        momentum: float = 0.98,
        epsilon: float = 1e-4,
        name: str = None,
        #dtype=None,
        #dynamic=False,
        trainable: bool = True,
    ) -> None:
        #super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.momentum = momentum
        self.epsilon = epsilon
        # Create layers
        #self.batch_norm = layers.BatchNormalization(
        #    axis=-1,
        #    momentum=momentum,
        #    epsilon=epsilon,
        #    trainable=trainable,
        #    name="BatchNorm",
        #)
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name="Conv2D",
            activation=activation,
            kernel_initializer=kernel_initializer,
        )
        #self.relu =layers.ReLU(
        #    name="ReLU",
        #) #<- lambda x:x
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
                "momentum": self.momentum,
                "epsilon": self.epsilon,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: #Training = True
        """
        x = self.conv(inputs)
        x = self.batch_norm(x,training=training)
        """
        # Previously BN -> CONV
        #x = self.batch_norm(inputs,training=training) 
        #x = self.conv(x)
        x = self.conv(inputs)
        #x = self.relu(x)
        #x = self.batch_norm(x,training=training) 
        return x
    def build(self, input_shape):
        self.built = True
