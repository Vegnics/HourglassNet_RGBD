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
            filters=8,
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
        return S
    def build(self, input_shape):
        super().build(input_shape)

@register_keras_serializable(package="lattentionSpatial") 
class SpatialEnergyHead(Layer):
    """
    This layer performs 2D convolution, Batch Normalization, and ReLU.
    """
    def __init__(
        self,
        K_size: int = 32,
        kernel_initializer: str = "glorot_uniform",
        name: str = None,
        trainable: bool = True,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.K_size = K_size
        self.kernel_initializer = kernel_initializer
        # Create layers
        #"""

        self.Vgen = layers.Dense(
                units=self.K_size,
                activation= "softmax", #None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                name = "Vgen_FC",
                input_shape=(16,)
                )
        
        self.Hgen = layers.Dense(self.K_size,
                activation= "softmax", #None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                name = "Hgen_FC",
                input_shape=(16,)
                )
        
        self.dropout_v = layers.Dropout(0.05)
        self.dropout_h = layers.Dropout(0.05)
        
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "kernel_initializer": self.kernel_initializer,
                "K_size": self.K_size,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        #gshape = tf.shape(sgap) #NHWC
        V = tf.reshape(self.dropout_v(self.Vgen(inputs),training=training),shape=(-1,self.K_size,1))
        H = tf.reshape(self.dropout_h(self.Hgen(inputs),training=training),shape=(-1,1,self.K_size))
        return tf.linalg.matmul(V,H)
    def build(self, input_shape):
        self.Vgen.build(input_shape)
        self.Hgen.build(input_shape)
        self.dropout_v.build((None, self.K_size))
        self.dropout_h.build((None, self.K_size))
        super().build(input_shape)
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
        headnum: int = 16,
        trainable: bool = True,
        kernel_reg: bool = False,
        feat_size: int = None
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
        self.feat_size = feat_size
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
            kernel_initializer= tf.constant_initializer(1/256.0),
            use_bias=False,
        )
        self.spatial_heads = [SpatialEnergyHead(K_size=self.feat_size,name=f"SpatialHead_{u}") for u in range(self.head_num)] 
        
        self.score_gen = layers.Conv2D(
            filters=1,
            kernel_size=(3,3),
            strides=(1,1),
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
                "kernel_reg": self.kernel_reg,
                "feat_size": self.feat_size
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        projection = self.gap_proj(inputs)
        K = tf.shape(inputs)[1]
        #_inputs = tf.reduce_mean(tf.math.square(inputs),keepdims=True,axis=-1)
        tiled = tf.reshape(projection, (-1, 4, K // 4, 4, K // 4))
        tiled = tf.transpose(tiled, perm=[0, 1, 3, 2, 4])
        energy_descriptor = tf.reshape(tf.reduce_mean(tf.math.square(tiled),axis=[3,4]),(-1,16))
        stacked_outs = tf.stack([self.spatial_heads[u](energy_descriptor) for u in range(self.head_num)],axis=-1)
        scores = tf.nn.sigmoid(self.score_gen(stacked_outs)) 
        return scores
    
    def build(self, input_shape):
        super().build(input_shape)
