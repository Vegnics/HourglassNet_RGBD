import tensorflow as tf
from keras import layers
from keras.layers import Layer
from hourglass_tensorflow.layers.dummy_layers import IdentityLayer
from keras.utils import register_keras_serializable

@register_keras_serializable(package="lConvBNRelu")
class ConvBatchNormReluLayer(Layer):
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
        momentum: float = 0.9,
        epsilon: float = 0.001,
        name: str = None,
        trainable: bool = True,
        use_relu: bool = True,
        normalized: bool = True,
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
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
        
        # Create layers

        self.batch_norm = layers.BatchNormalization(
            axis=-1,
            momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable,
            name="BatchNorm_Identity",
        ) if self.normalized else IdentityLayer("BatchNorm_Identity")
        
        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            name="Conv2D",
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
        )
        self.relu = layers.ReLU(
            name="ReLU_identity",
        ) if self.use_relu else IdentityLayer(name="ReLU_identity")

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
                "use_relu": self.use_relu,
                "normalized": self.normalized
            },
        }

    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        # Could it be CONV-> RELU -> BATCH NORM
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        return x 
    
    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """