import tensorflow as tf
from keras import layers
from keras.layers import Layer


class ConvReluBatchNormLayer(Layer):
    """
    This layer performs 2D convolution, ReLU, and finally Batch Normalization
    """
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        activation: str = None,
        kernel_initializer: str = "glorot_uniform",
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
        use_relu: bool = True,
        normalized: bool = True,
    ) -> None:
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        # Store Config
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_relu = use_relu
        self.normalized = normalized
        # Create Layers
        self.batch_norm = layers.BatchNormalization(
            axis=-1,
            momentum=momentum,
            epsilon=epsilon,
            trainable=trainable,
            name="BatchNorm",) if self.normalized else lambda x,**y:x
        
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name="Conv2D",
            activation=activation,
            kernel_initializer=kernel_initializer,
            ) 
        self.relu = layers.ReLU(name="ReLU",)  if self.use_relu else lambda x:x

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

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.conv(inputs)
        #x = self.batch_norm(x, training=training)
        #x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x, training=training)
        return x
    def build(self, input_shape):
        self.built = True