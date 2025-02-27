import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish
from keras.regularizers import L2,L1


class _SpatialBasedPooling(Layer):
    def __init__(
        self,
        filters: int,
        kernel_initializer: str = "glorot_uniform",
        name: str = None,
        trainable: bool = True,
    ) -> None:
        super().__init__(name=name,trainable=trainable)
        # Store config
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        # Create layers
        self.spatial_layers = [
            layers.Conv2D(
                filters=filters//4,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="AttConv2D1",
                activation="relu",
                #kernel_regularizer= RegL2(1e-5) if kernel_reg else None,
                kernel_initializer=kernel_initializer,
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2),
                padding="valid",
                name=f"AttFeaturelMaxPool1",
                trainable=trainable,
            ),

            layers.Conv2D(
                filters=filters,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="AttConv2D2",
                activation="relu",
                #kernel_regularizer= RegL2(1e-5) if kernel_reg else None,
                kernel_initializer=kernel_initializer,
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2),
                padding="valid",
                name=f"AttFeatureMaxPool2",
                trainable=trainable,
            )
        ]
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "filters": self.filters,
                "kernel_initializer": self.kernel_initializer
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        x = tf.cast(inputs,dtype=tf.float32)
        for layer in self.spatial_layers:
            x = layer(x)
        return x
    
    def build(self, input_shape):
        pass

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
        kernel_reg: bool = False,
    ) -> None:
        super().__init__(name=name,trainable=trainable)
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
        """
        self.heads = [
            layers.Dense(filters//4,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name = "Head_{}".format(i),
            bias_initializer='glorot_uniform',
            kernel_regularizer=L2(1e-5) if kernel_reg else None,
            #bias_regularizer=L2(1e-3)
        )
        for i in range(self.head_num)
        ]
        """

        # EXPERIMENTAL GAPP-FLATTEN
        self.spatialgap = _SpatialBasedPooling(filters)

        self.score_mha = layers.MultiHeadAttention(num_heads=8,key_dim=filters//4,value_dim=filters//4,dropout=0.08)

        self.pre_projection = layers.Dense(filters,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name = "PreProjection",
            bias_initializer='glorot_uniform',
            kernel_regularizer=L1(1e-5) if kernel_reg else None,
            #bias_regularizer=L2(1e-5)
        )
        """
        self.last_projection1 = layers.Dense(filters,
            activation="relu",
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name = "LastProjection",
            bias_initializer='glorot_uniform',
            kernel_regularizer=L1(1e-5) if kernel_reg else None,
            #bias_regularizer=L2(1e-5)
        )
        """

        self.last_projection = layers.Dense(filters,
            activation="softmax",
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name = "LastProjection",
            bias_initializer='glorot_uniform',
            kernel_regularizer=L1(1e-5) if kernel_reg else None,
            #bias_regularizer=L2(1e-5)
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
        _shape = tf.shape(inputs)
        tf.print("inputs shape:", _shape)
        H = _shape[1]
        W = _shape[2]
        #gap = tf.reduce_mean(inputs,axis=[1,2])
        learned_gap = self.spatialgap(inputs)
        learned_gap = tf.transpose(learned_gap,perm=[0,3,1,2])
        tf.print("learned_gap shape:", tf.shape(learned_gap))
        flatten_gap = tf.reshape(learned_gap,shape=(-1,256,(H//4)*(W//4)))
        _flatten_gap = self.pre_projection(flatten_gap) # (-1)
        #head_outs = []
        #for i in range (self.head_num):
        #    head_outs.append(self.heads[i](gap))
        #head_out = tf.concat(head_outs,axis=-1)
        mha_out = self.score_mha(_flatten_gap,_flatten_gap) #NCDim
        mha_out = tf.reshape(mha_out,shape=(-1,64*64))
        scores = self.last_projection(mha_out)
        #_out = self.last_projection1(_out)
        scores = tf.expand_dims(scores,axis=1)
        scores = tf.expand_dims(scores,axis=1)
        return scores
    def build(self, input_shape):
        pass
