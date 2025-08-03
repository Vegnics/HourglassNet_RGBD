from hashlib import file_digest
from matplotlib import axis
import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish
from keras.regularizers import L2 as RegL2
from keras.saving import register_keras_serializable
from keras import constraints
from keras import initializers
import keras
from hourglass_tensorflow.layers.sequential_layer import SequentialLayer
from positional_encodings.tf_encodings import TFPositionalEncoding2D 
from keras import ops

@register_keras_serializable(package="lattentionSpatial")
class ConvexComb(keras.constraints.Constraint):
  def __call__(self, w):
    w= tf.math.abs(w)
    return w/(tf.reduce_sum(w,axis=2,keepdims=True)+1e-7)
  
@register_keras_serializable(package="lattentionSpatial")
class ConvexComb1D(keras.constraints.Constraint):
  def __call__(self, w):
    w= tf.math.abs(w)
    return w/(tf.reduce_sum(w,keepdims=True)+1e-7)


@register_keras_serializable(package="lattentionSpatial")
class GeLU(Layer):
    def __init__(
            self,
            name: str = None
    ) -> None:
        super().__init__(name=name, trainable = False)
    def call(inputs):
        return tf.nn.gelu(inputs)

class SpatialSoftMax(Layer):
    def __init__(
            self,
            name: str = None
    ) -> None:
        super().__init__(name=name, trainable = False)
    def call(self,inputs ,training = False):
        shape = tf.shape(inputs)
        B = shape[0]
        K = shape[1]
        C = shape[3]
        flatten = tf.reshape(inputs,shape=(-1,K**2,C))
        flatten = tf.nn.softmax(flatten,axis=1)
        return tf.reshape(flatten,shape=(-1,K,K,C))

@register_keras_serializable(package="lattentionSpatial") 
class SpatialEnergyHead(Layer):
    """
    This layer generates a low-rank single-channel spatial representation.  
    """
    def __init__(
        self,
        K_size: int = 32,
        kernel_initializer: str = "glorot_uniform",
        name: str = None,
        trainable: bool = True,
        decoder: layers.Layer = None,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.K_size = K_size
        self.kernel_initializer = kernel_initializer
        # Create layers
        # Shared FC layer
        #self.shared_fc = layers.Dense(
        #        units=self.K_size//2,
        #        activation=None,
        #        use_bias=True,
        #        kernel_initializer='glorot_uniform',
        #        trainable=self.trainable
        #)
        #self.bn = layers.BatchNormalization(axis=-1,name="head_bn")

        #self.alpha = self.add_weight(
        #        shape=[1,],
        #        name="head_alpha",
        #        initializer=tf.constant_initializer(0.01),
        #        constraint=constraints.min_max_norm(min_value=0.001,max_value=0.1),
        #        trainable=self.trainable,
        #    )

        self.local = SequentialLayer(
           [
              layers.Conv2D(
                filters=1,
                kernel_size=(1,1),
                strides=(1,1),
                padding="same",
                name="head_local_conv",
                activation=None,
                kernel_regularizer= None,
                kernel_initializer= ConvexComb(), #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=False,
                trainable = self.trainable),

            layers.Conv2D(
                filters=16,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="head_local_conv",
                activation="sigmoid",
                kernel_regularizer= None,
                kernel_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable),

                #layers.LayerNormalization(axis=-1,name="ln_local",trainable=self.trainable),

           ],
           name="local_extractor"
        )

        self.extractor = SequentialLayer(
            [
                layers.Conv2D(
                filters=8,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="extractor0",
                activation="gelu",
                kernel_regularizer= None,
                kernel_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True),

                layers.DepthwiseConv2D(
                kernel_size=(3,3),
                strides=(2,2),
                padding="same",
                name="extractor1",
                activation="gelu",
                depthwise_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True),

                layers.MaxPooling2D(pool_size=(2,2),name="extractor_mp0"),

                layers.Conv2D(
                filters=16,
                kernel_size=(1,1),
                strides=(1,1),
                padding="same",
                name="extractor2",
                activation="sigmoid",
                kernel_regularizer= None,
                kernel_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable=self.trainable),

                layers.Dropout(0.1,name="head_dpout",noise_shape=(None,self.K_size//4,self.K_size//4,1)),

                #SpatialSoftMax(name="global_activ")

                #layers.LayerNormalization(axis=-1,name="extractor_ln",trainable=self.trainable),
                
            ],
            name="global_extractor",
            trainable=self.trainable
        )

        self.pos_encoding_xy = TFPositionalEncoding2D(32)

        #self.dropout_l = 
        self.upsample = layers.UpSampling2D(size=(4,4),interpolation="bilinear",name="head_upsample")
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "kernel_initializer": self.kernel_initializer,
                "K_size": self.K_size,
                "rank": self.rank,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        B = tf.shape(inputs)[0] 
        #lin_xy = tf.range(self.K_size)
        #X,Y = tf.meshgrid(lin_xy,lin_xy)
        #pos_encx = tf.tile(tf.expand_dims(self.pos_encoding(X),axis=0),(B,1,1,1))
        #pos_ency = tf.tile(tf.expand_dims(self.pos_encoding(Y),axis=0),(B,1,1,1))
        #_inputs = tf.concat([_inputs,pos_encx,pos_ency],axis=-1)
        #alphag = tf.reduce_sum(self.alpha)
        local_map = self.local(inputs,training=training) 
        #pos_enc = self.pos_encoding_xy(inputs)
        desc_posc = self.extractor(inputs) # (B,H,W,R)
        global_map = self.upsample(desc_posc)
        return global_map+local_map, tf.reshape(desc_posc,shape=(B,-1)) #0.5+tf.clip_by_value(0.58*tf.nn.tanh(0.7*att_map),-0.5,0.5) #tf.nn.sigmoid(att_map)# self.dropout_v(V,training=training),self.dropout_h(H,training=training)# (B,R,K),(B,R,K)  
    def build(self, input_shape):
        super().build(input_shape)

@register_keras_serializable(package="lattentionSpatial") 
class SpatialAttentionMechanism(Layer):
    """
    This layer generates spatial-wise raw attention scores from low-rank representations. 
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
        headnum: int = 4,
        trainable: bool = True,
        kernel_reg: bool = False,
        feat_size: int = None,
        rank: int = 6,
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
        self.rank = feat_size//2 #rank
        # Create layers

        #self.in_ln = layers.LayerNormalization(axis=-1,trainable=self.trainable, name="in_ln")

        self.head_mixer_w = self.add_weight(
                                    name="hmixerw",
                                    shape=(1,1,1,1,self.head_num),
                                    initializer="glorot_uniform",
                                    trainable=self.trainable,
                                    constraint=ConvexComb1D()
                                    )

        self.summarizer = SequentialLayer(
            [   
                layers.Conv2D(filters=32,
                            kernel_size=(1,1),
                            kernel_initializer="glorot_uniform",
                            name="summ_conv1",
                            activation= None, #"relu",
                            use_bias = False,
                            trainable = self.trainable 
                            ),
                layers.LayerNormalization(axis=-1,
                                          name="summ_ln",
                                          trainable=self.trainable),
                #keras.layers.Activation(activation= "gelu",name="summ_gelu")
            ],
            name="SAM_summarizer"
        )


        # Spatial Attention heads
        self.spatial_heads = [SpatialEnergyHead(K_size=self.feat_size,
                                                trainable=self.trainable,
                                                name=f"SpatialHead_{u}") 
                                                for u in range(self.head_num)] 
        

        # Set spatial heads as attributes to ensure full serialization.
        for k,layer in enumerate(self.spatial_heads):
            self.__setattr__(f"sam_{k}", layer)

        # Conv layer for combining the spatial heads' outputs.
        #"""
        self.score_gen = layers.Conv2D(
            filters=1,
            kernel_size=(1,1),
            strides=(1,1),
            padding="same",
            name="AttConv2D_scores",
            activation=None,
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= "zeros",#initializers.RandomNormal(mean=0.0, stddev=0.001),
            bias_initializer=tf.constant_initializer(0.0),
            use_bias=True,
            trainable = self.trainable
        )

        #self.softmax_layer = layers.Softmax(axis=[1,2])
        #"""
        
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
                "feat_size": self.feat_size,
                "rank":self.rank
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        # Compute the mean energy 2D 4x4 tile. 
        #_inputs = tf.clip_by_value(inputs,-1e4,1e4)
        #projection = self.gap_proj(_inputs)
        #_inputs = self.in_ln(inputs)
        B = tf.shape(inputs)[0] 
        K = self.feat_size # Width or Height
        spatial_desc = self.summarizer(inputs)
        heads_outs = [self.spatial_heads[u](spatial_desc,training=training) for u in range(self.head_num)]
        #stacked_outs = tf.concat([heads_outs[u][0] for u in range(self.head_num)],axis=-1)
        stacked_outs = tf.stack([heads_outs[u][0] for u in range(self.head_num)],axis=-1) #(B,H,W,D,Heads)
        stacked_outs_w = self.head_mixer_w*stacked_outs
        outs_aggregated = tf.reduce_sum(stacked_outs_w,axis=-1) #(B,H,W,D)

        #stacked_latent = tf.stack([heads_outs[u][1] for ua in range(self.head_num)],axis=-1)
        #scores =  0.1+tf.clip_by_value(0.95*tf.nn.sigmoid((self.score_gen(outs_aggregated)))-0.05,0.0,0.9)
        scores =  tf.clip_by_value(1.0+0.5*tf.nn.tanh((self.score_gen(outs_aggregated))),0.1,1.5)


        # Enforce higher variance at the heads' features
        latent_mean = tf.reduce_mean(stacked_outs,axis=-1,keepdims=True)
        latent_var = tf.reduce_mean(tf.square(stacked_outs-latent_mean),axis=-1)
        var_reg = tf.reduce_mean(1/tf.maximum(latent_var,0.001))
        self.add_loss(1e-6*var_reg)
        return scores #scores_raw # (B,K,K,1)=(B,H,W,1)
    
    def build(self, input_shape):
        super().build(input_shape)
