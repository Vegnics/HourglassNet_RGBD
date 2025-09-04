from calendar import c
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


@register_keras_serializable(package="lattentionFeature")
class FeatureAttentionHead(Layer):
    def __init__(
        self,
        filters: int,
        trainable: bool = True,
        kernel_reg: bool = False,
        name: str = None,
    ) -> None:
        super().__init__(name=name,trainable=trainable)
        self.filters = filters
        self.trainable = trainable
        self.kernel_reg = kernel_reg

        self.layer_seq = SequentialLayer(
            layer_list = [    
                layers.Dense(self.filters//8,
                    activation= "gelu", #None,
                    use_bias=True,
                    bias_initializer= "zeros", #tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=L2(1e-6) if self.kernel_reg else None,
                    name = "head_dense_A",
                    ),
                layers.Dense(self.filters//4,
                    activation= "gelu", #None,
                    use_bias=True,
                    bias_initializer= "zeros",#tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=L2(1e-6) if self.kernel_reg else None,
                    name = "head_dense_B",
                    ),
                #layers.LayerNormalization(axis=-1,name="head_LN")
                ],
            name = "FeatureAttentionHead",
            trainable=self.trainable
            )
        
    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        energy_tensor = tf.reduce_mean(tf.math.square(inputs),axis=[1,2])# per-channel mean energy
        x = self.layer_seq(energy_tensor,training=training)
        return x
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "filters": self.filters,
                "kernel_reg":self.kernel_reg,
            },
        }

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
        outmax: float = 1.0,
        name: str = None,
        headnum: int = 4,
        trainable: bool = True,
        kernel_reg: bool = False,
    ) -> None:
        super().__init__(name=name,trainable=trainable)
        # Store config
        self.filters = filters
        #self.kernel_size = kernel_size
        #self.strides = strides
        #self.padding = padding
        #self.activation = activation
        #self.kernel_initializer = kernel_initializer
        #self.momentum = momentum
        #self.epsilon = epsilon
        self.outmax = outmax
        self.head_num = headnum
        self.trainable = trainable
        self.kernel_reg = kernel_reg
        # Create layers

        #self.norm_layer = layers.LayerNormalization(axis=-1,
        #                                       epsilon=0.0001,
        #                                       name="main_LN"
        #                                       )
        self.in_ln = layers.LayerNormalization(axis=-1,trainable=self.trainable, name="in_ln")
        self.out_ln = layers.LayerNormalization(axis=-1,trainable=self.trainable, name="out_ln")

        # Feature Attention heads (FC->FC->LN)
        self.heads = [
                FeatureAttentionHead(self.filters,
                                     trainable=self.trainable,
                                     name = "Head_{}".format(i))
            for i in range(self.head_num)
        ]

        # Set all Attention heads as attributes to ensure full serialization.
        for k,layer in enumerate(self.heads):
            self.__setattr__(f"fam_{k}", layer)

        # Dropout previous to the raw score generation layer
        #self.dropout_last = layers.Dropout(0.08)

        # Raw score generation layer (aka Last projection)
        self.last_projection = SequentialLayer(
            layer_list=[
                #layers.LayerNormalization(axis=-1,name="LN_lastproj"),
                layers.Dense(self.filters//8,
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
                            kernel_initializer= "glorot_uniform",#initializers.RandomNormal(mean=0.0, stddev=0.001), #"glorot_normal",#initializers.Constant(value=1/float(self.filters)),
                            kernel_regularizer=L1(1e-5) if self.kernel_reg else None,
                            name = "lasproj_dense_B"
                        ),
                    ],
            name = "LastProjection",
            trainable=self.trainable
        )
        
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "filters": self.filters,
                #"kernel_size": self.kernel_size,
                #"strides": self.strides,
                #"padding": self.padding,
                #"activation": self.activation,
                #"kernel_initializer": self.kernel_initializer,
                #"momentum": self.momentum,
                #"epsilon": self.epsilon,
                "outmax":self.outmax,
                "headnum":self.head_num,
                "kernel_reg":self.kernel_reg,
            },
        }
    
    def call(self, inputs, y_true=None, training=False):
    #def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        # Mean Energy tensor computation 
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        stack_out = tf.ones(shape=[batch_size, height, width, 4], dtype=inputs.dtype)

        _inputs = self.in_ln(inputs,training=training)
        splitted = tf.split(_inputs, num_or_size_splits=self.head_num, axis=-1) # B,H,W,filters/self.head_num
        head_outs = [self.heads[i](splitted[i],training=training) for i in range(self.head_num)]
        # Input the Mean Energy Tensor to the Feature Attention heads 
        head_outs = []
        for i in range (self.head_num):
            head_outs.append(self.heads[i](splitted[i],training=training)) # B,filters/self.head_num
        head_out = tf.concat(head_outs,axis=-1)
        
        _head_out = self.in_ln(head_out)
        scores_raw = self.last_projection(_head_out)
        scores_raw = tf.expand_dims(scores_raw,axis=1)
        scores_raw = tf.expand_dims(scores_raw,axis=1)
        scores = tf.nn.sigmoid(scores_raw)#1.0 + 0.5*tf.nn.tanh(scores_raw)

        return scores,stack_out #scores_raw # The raw scores are input to an activation function f: R -> (0,1)   
    def build(self, input_shape):
        super().build(input_shape)
