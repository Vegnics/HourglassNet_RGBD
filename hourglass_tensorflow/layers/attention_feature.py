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
        #"""
        #self.heads = []
        # EXPERIMENTAL GAPP-FLATTEN
        self.norm_layer = layers.LayerNormalization(axis=-1,
                                               epsilon=0.0001,
                                               name="main_LN"
                                               )
                                               
        self.heads = [
                SequentialLayer(
                    layer_list = [
                    layers.Dense(self.filters//16,
                        activation= "gelu", #None,
                        use_bias=True,
                        bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=L2(1e-6) if self.kernel_reg else None,
                        name = "head_dense_A",
                        ),
                    layers.Dense(self.filters//4,
                        activation= "gelu", #None,
                        use_bias=True,
                        bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
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

        for k,layer in enumerate(self.heads):
            self.__setattr__(f"fam_{k}", layer)

        # EXPERIMENTAL GAPP-FLATTEN
        #self.spatialgap = _SpatialBasedPooling(self.filters)
        self.dropout_last = layers.Dropout(0.05)
        self.last_projection = SequentialLayer(
            layer_list=[
                layers.Dense(self.filters//16,
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
                            bias_initializer=initializers.Constant(value=0.0),
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
        #gap = tf.math.sqrt(tf.reduce_mean(tf.math.square(inputs),axis=[1,2])+1e-9)
        _shape = tf.shape(inputs)
        #_clip_inputs = tf.clip_by_value(inputs,-1e6,1e6)
        #tf.print("inputs shape:", _shape)
        #learned_gap = tf.clip_by_value(tf.reduce_mean(tf.math.square(_clip_inputs),axis=[1,2]),0,1e8)
        learned_gap = tf.reduce_mean(tf.math.square(inputs),axis=[1,2])

        #learned_gap = self.spatialgap(inputs)
        #learned_gap = tf.reduce_mean(inputs,axis=[1,2]) #NC
        learned_gap = tf.reshape(learned_gap,shape=(-1,self.filters))
        head_outs = []
        for i in range (self.head_num):
            head_outs.append(self.heads[i](learned_gap))
        head_out = tf.concat(head_outs,axis=-1)
        head_stack = tf.stack(head_outs,axis=-1) #B,d_out,N_head
        head_mean = tf.reduce_mean(head_stack,keepdims=True,axis=-1)
        head_var = tf.reduce_mean(tf.math.square(head_stack-head_mean),axis=[1,2])+1e-6
        penalty = tf.reduce_mean(tf.math.sqrt(head_var))
        loss_diversity = tf.nn.softplus(1.5 - penalty)  # prefer variance > 1.5
        #self.add_loss(λ * penalty)
        #loss_diversity = tf.exp(-10.0 * tf.clip_by_value(head_var, 0.0, 5.0))
        #loss_diversity = tf.math.exp(-1.0*head_var)  # penalize low diversity
        self.add_loss(0.00001 * loss_diversity)
        #_head_out = tf.nn.relu(head_out)
        #_head_out = self.norm_layer(_head_out)
        _head_out = self.dropout_last(head_out,training=training)
        scores = self.last_projection(_head_out)
        #scores = tf.nn.sigmoid(scores)
        #scores_shape = tf.shape(scores)
        #alpha_batch = 4.0*self.alpha*tf.ones(shape=(scores_shape[0],1))
        #scoreswalpha = tf.concat([scores,alpha_batch],axis=-1)
        scores = tf.expand_dims(scores,axis=1)
        scores_raw = tf.expand_dims(scores,axis=1)
        return scores_raw #,
    def build(self, input_shape):
        super().build(input_shape)
