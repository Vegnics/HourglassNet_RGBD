import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish
from keras.regularizers import L2 as RegL2
from keras.saving import register_keras_serializable

@register_keras_serializable(package="lattentionSpatial") 
class SpatialAttentionMechanism(Layer):
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
        trainable: bool = True,
        kernel_reg: bool = False,
    ) -> None:
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
        self.outmax = outmax
        self.head_num = headnum
        self.trainable = trainable
        self.kernel_reg = kernel_reg
        # Create layers
        #"""
        self.gap_proj = layers.Conv2D(
            filters=1,
            kernel_size=(1,1),
            strides=self.strides,
            padding="same",
            name="proj_gap",
            activation=None,
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= self.kernel_initializer,
            use_bias=False,
        )
        self.bn1 = layers.BatchNormalization(
            axis=-1,
            momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable,
            name="BN_gap",
        )
        #"""

        self.conv1_3x3 = layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            strides=self.strides,
            padding="same",
            name="AttConv2D_1",
            activation= None, #"gelu",
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= self.kernel_initializer,
            use_bias=False,
        )

        self.bn2 = layers.BatchNormalization(
            axis=-1,
            momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable,
            name="BN_conv1",
        )

        self.maxpool1 = layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="valid",
            name=f"AttSpatialMaxPool1",
            trainable=self.trainable,
        )

        self.conv2_3x3 = layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            strides=self.strides,
            padding="same",
            name="AttConv2D_2",
            activation= None,#"gelu",
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= self.kernel_initializer,
            use_bias=False,
        )

        self.bn3 = layers.BatchNormalization(
            axis=-1,
            momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable,
            name="BN_conv2",
        )
        
        self.maxpool2 = layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="valid",
            name=f"AttSpatialMaxPool2",
            trainable=self.trainable,
        )

        self.patches_proj = layers.Conv2D(
            filters=16,
            kernel_size=(1,1),
            strides=self.strides,
            padding="same",
            name="AttConv2D_patch",
            activation=None,
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= self.kernel_initializer,
        )

        #self.bn4 = layers.BatchNormalization(
        #    axis=-1,
        #    momentum=self.momentum,
        #    epsilon=self.epsilon,
        #    trainable=trainable,
        #    name="BN_patch",
        #)

        self.score_gen = layers.Conv2D(
            filters=1,
            kernel_size=(3,3),
            strides=self.strides,
            padding="same",
            name="AttConv2D_scores",
            activation="sigmoid",
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= "zeros",
            use_bias=False,
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
                "outmax": self.outmax,
                "headnum": self.head_num,
                "kernel_reg": self.kernel_reg
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        gap = tf.reduce_mean(inputs,axis=-1)
        gap = tf.expand_dims(gap,axis=-1)
        learned_gap = self.gap_proj(inputs)
        sgap = self.bn1(gap + learned_gap,training=training)
        sgap = tf.nn.swish(sgap)
        gshape = tf.shape(sgap) #NHWC
        S = self.conv1_3x3(sgap)
        S = self.bn2(S,training=training)
        S = tf.nn.swish(S)

        S = self.maxpool1(S)
        S = self.conv2_3x3(S)
        S = self.bn3(S,training=training)
        S = tf.nn.swish(S)

        S = self.maxpool2(S)
        S = self.patches_proj(S)
        #S = self.bn4(S,training=training)
        #S = tf.nn.swish(S)

        # Ensure H and W are divisible by 4 before reshaping
        H, W = gshape[1], gshape[2]
        # Reshape channels into spatial patches (assuming 16 channels split into 4x4)
        S = tf.reshape(S, shape=(-1, H//4,W //4, 4, 4))  # Shape: (B, H/4, 4, W/4, 4, 4)
        S = tf.transpose(S, perm=[0, 1, 3, 2, 4])  # Swap inner spatial blocks
        # Merge the new spatial structure back into H and W
        S = tf.reshape(S, shape=(-1, H, W, 1))  # Final shape (B, H, W, C)
        scores = self.score_gen(S) 
        return scores-0.5
    
    def build(self, input_shape):
        super().build(input_shape)
