from turtle import pen
from numpy import expand_dims
import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish
from keras.regularizers import L2,L1
from keras.saving import register_keras_serializable
from keras import initializers
import keras
from hourglass_tensorflow.layers.sequential_layer import SequentialLayer

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
        self.spatial_layers = []
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
        self.spatial_layers = [
        layers.Conv2D(
            filters=self.filters//8,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            name="AttConv2D1",
            activation="gelu",
            kernel_initializer=self.kernel_initializer,
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="valid",
            name=f"AttFeaturelMaxPool1",
        ),

        layers.Conv2D(
            filters=self.filters,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            name="AttConv2D2",
            activation="gelu",
            kernel_initializer=self.kernel_initializer,
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="valid",
            name=f"AttFeatureMaxPool2",
        )
        ]
        super().build(input_shape)
        self.built = True

@register_keras_serializable(package="lattentionFeature") 
class FeatureAttentionMechanism(Layer):
    """
    This layer regards the Mean Energy Tensor (MET) of the input features to generate
    raw channel-wise scores. It employs multiple parallel Feature Attention Heads
    to process the MET, concatenate the heads outputs into a single 1d tensor and use 
    2 FC layers to compute the raw scores. Head output orthogonality regularization is 
    applied.   
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
        headnum: int = 5,
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
        self.trainable = trainable
        self.kernel_reg = kernel_reg
        # Create layers

        #self.norm_layer = layers.LayerNormalization(axis=-1,
        #                                       epsilon=0.0001,
        #                                       name="main_LN"
        #                                       )
        self.in_bn = layers.BatchNormalization(axis=-1,trainable=self.trainable)

        # Feature Attention heads (FC->FC->LN)
        self.heads = [
                SequentialLayer(
                    layer_list = [
                    layers.Dense(self.filters//16,
                        activation= "gelu", #None,
                        use_bias=True,
                        bias_initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=L2(1e-6) if self.kernel_reg else None,
                        name = "head_dense_A",
                        ),
                    layers.Dense(self.filters//4,
                        activation= "gelu", #None,
                        use_bias=True,
                        bias_initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=L2(1e-6) if self.kernel_reg else None,
                        name = "head_dense_B",
                        ),
                    layers.LayerNormalization(axis=-1,name="head_LN")
                    ],
                name = "Head_{}".format(i),
                trainable=self.trainable
                )
            for i in range(self.head_num)
        ]

        # Set all Attention heads as attributes to ensure full serialization.
        for k,layer in enumerate(self.heads):
            self.__setattr__(f"fam_{k}", layer)

        # Dropout previous to the raw score generation layer
        self.dropout_last = layers.Dropout(0.08)

        # Raw score generation layer (aka Last projection)
        self.last_projection = SequentialLayer(
            layer_list=[
                layers.Dense(self.filters//4,
                            activation="gelu",
                            use_bias=True,
                            bias_initializer="zeros",
                            kernel_initializer="glorot_uniform",
                            kernel_regularizer=L1(1e-5) if self.kernel_reg else None,
                            name = "lasproj_dense_A"
                        ),
                layers.Dense(self.filters,
                            activation=None,
                            use_bias=True,
                            bias_initializer="zeros",
                            kernel_initializer="glorot_normal",#initializers.Constant(value=1/float(self.filters)),
                            kernel_regularizer=L1(1e-5) if self.kernel_reg else None,
                            name = "lasproj_dense_B"
                        )
                    ],
            name = "LastProjection",
            trainable=self.trainable
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
                "outmax":self.outmax,
                "headnum":self.head_num,
                "kernel_reg":self.kernel_reg,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        # Mean Energy tensor computation 
        _shape = tf.shape(inputs)
        _inputs =self.in_bn(inputs,training=training) # Batch normalization to the raw inputs
        _inputs = tf.clip_by_value(_inputs,-1e4,1e4) # Clip the input feats to avoid numerical instability
        #energy_tensor = tf.math.sqrt(tf.reduce_mean(tf.math.square(_inputs),axis=[1,2])+1e-6)# per-channel mean energy
        energy_tensor = tf.reduce_mean(tf.math.square(_inputs),axis=[1,2])# per-channel mean energy  

        # Input the Mean Energy Tensor to the Feature Attention heads 
        head_outs = []
        for i in range (self.head_num):
            head_outs.append(self.heads[i](energy_tensor))
        head_out = tf.concat(head_outs,axis=-1)
        
        # Heads' output orthogonality regularization
        #head_stack = tf.stack(head_outs,axis=-1) # Stacking heads' output -> B,d_out,N_head
        #normed_heads = tf.nn.l2_normalize(head_stack, axis=1) # along d_head (B,d_head,N_heads)
        #similarity = tf.matmul(tf.transpose(normed_heads,perm=[0,2,1]), normed_heads)  # cosine similarity between head's outputs (B,N_heads,N_heads)
        #mask = 1.0 - tf.eye(self.head_num)
        #penalty = tf.reduce_mean(tf.square(similarity * mask),axis=[1,2])
        #penalty = tf.reduce_mean(tf.math.sqrt(penalty+1e-6))
        #self.add_loss(1e-4 * penalty)
        
        # Raw scores computation -> (B,1,1,C) 
        #_head_out = self.norm_layer(_head_out)
        _head_out = self.dropout_last(head_out,training=training)
        scores_raw = self.last_projection(_head_out)
        scores_raw = tf.expand_dims(scores_raw,axis=1)
        scores_raw = tf.expand_dims(scores_raw,axis=1)
        scores = 1.0 + 0.9*tf.nn.tanh(scores_raw)

        #scores = tf.clip_by_value(scores_raw,-2.2,2.2)
        #scores = (tf.nn.sigmoid(scores)-tf.nn.sigmoid(-2.2))/(tf.nn.sigmoid(2.2)-tf.nn.sigmoid(-2.2))

        scores_mean = tf.reduce_mean(scores,axis=-1,keepdims=True)
        scores_var = tf.reduce_mean(tf.square(scores-scores_mean),axis=-1)
        var_reg = tf.reduce_mean(1/tf.maximum(scores_var,0.001))
        self.add_loss(1e-6*var_reg)
        return scores #scores_raw # The raw scores are input to an activation function f: R -> (0,1)   
    def build(self, input_shape):
        super().build(input_shape)
