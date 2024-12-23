import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish
from keras.regularizers import L2,L1


class FeatureAttentionMechanism(Layer):
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
        epsilon: float = 1e-3,
        outmax: float = 1.0,
        name: str = None,
        headnum: int = 8,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
    ) -> None:
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        # Store config
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.momentum = momentum
        self.epsilon = epsilon
        self.outmax = outmax
        self.head_num = headnum
        # Create layers

        self.heads = [
            layers.Dense(filters//4,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name = "Head_{}".format(i),
            bias_initializer='glorot_uniform',
            kernel_regularizer=L1(1e-4),
            bias_regularizer=L2(1e-3)
        )
        for i in range(self.head_num)
        ]

        self.last_projection = layers.Dense(filters,
            activation="sigmoid",
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name = "LastProjection",
            bias_initializer='glorot_uniform',
            kernel_regularizer=L2(1e-6),
            bias_regularizer=L2(1e-5)
        )
        
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

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        #gap = tf.math.sqrt(tf.reduce_mean(tf.math.square(inputs),axis=[1,2])+1e-9)
        gap = tf.reduce_mean(inputs,axis=[1,2])
        head_outs = []
        for i in range (self.head_num):
            head_outs.append(self.heads[i](gap))
        head_out = tf.concat(head_outs,axis=-1)
        scores = self.last_projection(head_out)
        scores = tf.expand_dims(scores,axis=1)
        scores = tf.expand_dims(scores,axis=1)
        return (0.0001+scores)*inputs
    def build(self, input_shape):
        pass
