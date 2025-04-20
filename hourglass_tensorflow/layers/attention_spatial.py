import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish
from keras.regularizers import L2 as RegL2
from keras.saving import register_keras_serializable
from keras import constraints

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
                activation= None, #None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                name = "Vgen_FC",
                )
        
        #self.diff_vec = self.add_weight(
        #        shape=[1,16],
        #        name="diff_vec",
        #        initializer="glorot_normal",
        #        trainable=self.trainable,
        #    )

        self.Hgen = layers.Dense(self.K_size,
                activation=None, #None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                name = "Hgen_FC",
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
        #input_v = tf.transpose(tf.stack((inputs,inputs-self.diff_vec),axis=-1),perm=[0,2,1])
        #input_h = tf.transpose(tf.stack((inputs,inputs+self.diff_vec),axis=-1),perm=[0,2,1])
        _input = tf.transpose(inputs,perm=[0,2,1]) #(B,3,K)
        #input_h = tf.transpose(tf.stack((inputs,inputs+self.diff_vec),axis=-1),perm=[0,2,1])
        v = tf.nn.relu(self.Vgen(_input))
        h = tf.nn.relu(self.Hgen(_input ))   
        V = self.dropout_v(tf.math.l2_normalize(v), training=training)  # (B,3,K)
        H = self.dropout_h(tf.math.l2_normalize(h), training=training)  # (B,3,K)
        V = tf.transpose(V,perm=[0,2,1])                # (B,K,3)
        #H = tf.reshape(H, (-1, 2, self.K_size))                # (B,2,K)

        return tf.matmul(V, H)                            # (B,K,K)
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
            kernel_constraint = constraints.NonNeg(),
            kernel_initializer= tf.constant_initializer(1/256.0),
            use_bias=False,
        )
        self.bn = layers.BatchNormalization(axis=-1,
                                            momentum=0.9,
                                            trainable=self.trainable,
                                            name="BN_sam")
        
        self.ln = layers.LayerNormalization(axis=-1,
                                            trainable=self.trainable,
                                            name="LN_sam")

        self.spatial_heads = [SpatialEnergyHead(K_size=self.feat_size,name=f"SpatialHead_{u}") for u in range(self.head_num)] 
        
        for k,layer in enumerate(self.spatial_heads):
            self.__setattr__(f"sam_{k}", layer)

        self.score_gen = layers.Conv2D(
            filters=1,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            name="AttConv2D_scores",
            activation=None,
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= "glorot_uniform",
            #bias_initializer=tf.constant_initializer(-3.0),
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
        _inputs = self.bn(inputs,training=training)
        projection = self.gap_proj(_inputs)
        K = tf.shape(inputs)[1]
        #_inputs = tf.reduce_mean(tf.math.square(inputs),keepdims=True,axis=-1)
        tiled = tf.reshape(projection, (-1, 4, K // 4, 4, K // 4))
        tiled = tf.transpose(tiled, perm=[0, 1, 3, 2, 4])
        tgrid = tf.range(0,4,1,dtype=tf.float32)
        X,Y = tf.meshgrid(tgrid,tgrid)
        energy_tile = tf.reduce_mean(tf.math.square(tiled),axis=[3,4])
        pos_encoding = tf.repeat(tf.stack([X,Y],axis=-1),repeats=tf.shape(inputs)[0],axis=0)
        energy_descriptor = tf.reshape(tf.concat([energy_tile,pos_encoding],axis=-1),(-1,16,3))
        stacked_outs = tf.stack([self.spatial_heads[u](energy_descriptor,training=training) for u in range(self.head_num)],axis=-1)
        stacked_outs = self.ln(stacked_outs)
        scores = tf.clip_by_value(tf.nn.swish(self.score_gen(stacked_outs)),0.0,1.0) 
        return scores
    
    def build(self, input_shape):
        super().build(input_shape)
