import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish
from keras.regularizers import L2 as RegL2
from keras.saving import register_keras_serializable

@register_keras_serializable(package="lattentionSpatial") 
class SpatialHead(Layer):
    """
    This layer performs 2D convolution, Batch Normalization, and ReLU.
    """
    def __init__(
        self,
        filters: int = 32,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "same",
        activation: str = None,
        kernel_initializer: str = "glorot_uniform",
        momentum: float = 0.9,
        epsilon: float = 1e-3,
        name: str = None,
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
        self.trainable = trainable
        self.kernel_reg = kernel_reg
        # Create layers
        #"""
        self.conv1_3x3 = layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            strides= (1,1),#self.strides,
            padding="same",
            name="AttConv2D_1",
            activation= None, #"gelu",
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= self.kernel_initializer,
            use_bias=False,
        )

        self.ln_1 = layers.LayerNormalization(
            axis=-1,
            #momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable,
            name="LN_conv1",
        )

        self.maxpool1 = layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="valid",
            name=f"AttSpatialMaxPool1",
            trainable=self.trainable,
        )

        self.conv2_3x3 = layers.Conv2D(
            filters=16,
            kernel_size=(3,3),
            strides= (1,1),#self.strides,
            padding="same",
            name="AttConv2D_2",
            activation= None,#"gelu",
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= self.kernel_initializer,
            use_bias=False,
        )

        self.ln_2 = layers.LayerNormalization(
            axis=-1,
            #momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable,
            name="LN_conv2",
        )
        
        self.maxpool2 = layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="valid",
            name=f"AttSpatialMaxPool2",
            trainable=self.trainable,
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
                "kernel_reg": self.kernel_reg
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        #gshape = tf.shape(sgap) #NHWC
        S = self.conv1_3x3(inputs)
        S = self.ln_1(S)
        S = tf.nn.relu(S)
        S = self.maxpool1(S)

        S = self.conv2_3x3(S)
        S = self.ln_2(S)
        S = tf.nn.relu(S)
        S = self.maxpool2(S)
        return S
    def build(self, input_shape):
        super().build(input_shape)

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
            use_bias=True,
        )
        self.spatial_heads = [[SpatialHead(name="SpatialHead_{u}_{v}") for v in range(4)] for u in range(4)] 
        
        self.score_gen = layers.Conv2D(
            filters=1,
            kernel_size=(3,3),
            strides=self.strides,
            padding="same",
            name="AttConv2D_scores",
            activation=None,
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= "zeros",
            bias_initializer=tf.constant_initializer(-3.0),
            use_bias=True,
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
        learned_gap = self.gap_proj(inputs)
        row_outputs = []
        for u in range(4):
            col_outputs = []
            for v in range(4):
                head_out = self.spatial_heads[u][v](learned_gap, training=training)  # (B, H', W', C)
                col_outputs.append(head_out)
            row_outputs.append(tf.concat(col_outputs, axis=2))  # concat along width
        stacked_heads = tf.concat(row_outputs, axis=1)  # concat along height → (B, H_total, W_total, C)
        scores = tf.nn.sigmoid(self.score_gen(stacked_heads)) 
        return scores
    
    def build(self, input_shape):
        super().build(input_shape)
