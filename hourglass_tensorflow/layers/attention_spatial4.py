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
  def __call__(self, w,dtype=None):
    w= tf.math.abs(w)
    return w/(tf.reduce_sum(w,axis=2,keepdims=True)+1e-7)
  
@register_keras_serializable(package="lattentionSpatial")
class ConvexComb1D(keras.constraints.Constraint):
  def __call__(self, w,dtype=None):
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
class _GlobalPathLayer(Layer):
    def __init__(
            self,
            name: str = None
    ) -> None:
        super().__init__(name=name, trainable = False)
        """
        self.downskip_1 = layers.Conv2D(
                filters= 16,
                kernel_size=(3,3),
                strides=(2,2),
                padding="same",
                name="downskip_1",
                activation="gelu",
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable)
        """
        self.global_extractor = layers.Conv2D(
                filters=16,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="extractor",
                activation="gelu",
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable)
        
        self.pos_encoding_xy = TFPositionalEncoding2D(16)

        self.downskip_1 = layers.MaxPool2D(pool_size=(2,2),name="maxpool1")
        
        self.core_1 = layers.Conv2D(
                filters= 8,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="core_1",
                activation="gelu",
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable)
        
        self.core_1_down = layers.Conv2D(
                filters= 16,
                kernel_size=(3,3),
                strides=(2,2),
                padding="same",
                name="core_1_down",
                activation="gelu",
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable)
        
        self.downskip_2 = layers.MaxPool2D(pool_size=(2,2),name="maxpool2")
        
        """
        self.downskip_2 = layers.Conv2D(
                filters= 16,
                kernel_size=(3,3),
                strides=(2,2),
                padding="same",
                name="downskip_1",
                activation="gelu",
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable)
        """
        
        self.core_2 = layers.Conv2D(
                filters= 8,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="core_2",
                activation="gelu",
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable)
        
        self.core_2_down = layers.Conv2D(
                filters= 16,
                kernel_size=(3,3),
                strides=(2,2),
                padding="same",
                name="core_2_down",
                activation="gelu",
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable)
        
        self.upsample_deconv = layers.Conv2DTranspose(
                filters= 16,
                kernel_size=(5,5),
                strides=(4,4),
                padding="same",
                name="upsample",
                activation="gelu",
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable)
        
        self.score_gen = layers.Conv2D(
            filters=1,
            kernel_size=(1,1),
            strides=(1,1),
            padding="same",
            name="global_scores",
            activation="sigmoid",
            kernel_initializer= "glorot_uniform",#initializers.RandomNormal(mean=0.0, stddev=0.001),
            bias_initializer=tf.constant_initializer(0.0),
            use_bias=True,
            trainable = self.trainable
        )
        
    def call(self,inputs ,training = False):
        _inputs = self.global_extractor(inputs)
        pos_encoding = self.pos_encoding_xy(_inputs)
        _inputs = _inputs + pos_encoding
        skip1 = self.downskip_1(_inputs)
        x = self.core_1(_inputs)
        x = self.core_1_down(x)+skip1
        skip2 = self.downskip_2(x)
        x = self.core_2(x)
        x = self.core_2_down(x) + skip2
        return self.score_gen(self.upsample_deconv(x))
    
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

        """
        self.summarizer = layers.Conv2D(
                filters=4,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="summarizer",
                activation=None,
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=False,
                trainable = self.trainable)
        """


        """
            layers.Conv2D(
                filters=16,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="head_local_conv_2",
                activation="gelu",
                kernel_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable),
            """
        self.local_path = SequentialLayer(
           [
              layers.Conv2D(
                filters=16,#8,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="head_local_conv_1",
                activation="gelu",
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable)
            

                #layers.LayerNormalization(axis=-1,name="ln_local",trainable=self.trainable),

           ],
           name="local_extractor"
        )

        self.local_path2 = SequentialLayer(
           [  
                layers.MaxPool2D(pool_size=(2,2),name="mpool_0"),
                
                layers.Conv2D(
                    filters=16,#8,
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding="same",
                    name="head_local_conv_2",
                    activation="gelu",
                    kernel_initializer = "glorot_uniform",
                    bias_initializer=tf.constant_initializer(0.0),
                    use_bias=True,
                    trainable = self.trainable),

                layers.UpSampling2D(size=(2,2),interpolation="bilinear",name="upsample")
                #layers.LayerNormalization(axis=-1,name="ln_local",trainable=self.trainable),

           ],
           name="local_extractor2"
        )

        self.score_gen = layers.Conv2D(
            filters=1,
            kernel_size=(1,1),
            strides=(1,1),
            padding="same",
            name="HeadConv2D_scores",
            activation="sigmoid",
            kernel_initializer= "he_normal",#initializers.RandomNormal(mean=0.0, stddev=0.001),
            bias_initializer=tf.constant_initializer(0.0),
            use_bias=True,
            trainable = self.trainable
        )
        
        """
        self.global_path = SequentialLayer(
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

                layers.Conv2D(
                filters=8,
                kernel_size=(3,3),
                strides=(2,2),
                padding="same",
                name="extractor1",
                activation="gelu",
                kernel_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True),

                layers.DepthwiseConv2D(
                kernel_size=(3,3),
                strides=(2,2),
                padding="same",
                name="extractor2",
                activation="gelu",
                depthwise_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True),

                #layers.MaxPooling2D(pool_size=(2,2),name="extractor_mp0"),

                layers.Conv2D(
                filters=16,
                kernel_size=(1,1),
                strides=(1,1),
                padding="same",
                name="extractor3",
                activation="gelu",
                kernel_regularizer= None,
                kernel_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable=self.trainable),

                #layers.Dropout(0.1,name="head_dpout",noise_shape=(None,self.K_size//4,self.K_size//4,1)),

                #SpatialSoftMax(name="global_activ")

                #layers.LayerNormalization(axis=-1,name="extractor_ln",trainable=self.trainable),
                
            ],
            name="global_extractor",
            trainable=self.trainable
        )
        """
        #self.global_path = _GlobalPathLayer(name="global_path")

        #self.pos_encoding_xy = TFPositionalEncoding2D(4)

        #self.dropout_l = 
        #self.upsample = layers.UpSampling2D(size=(4,4),interpolation="bilinear",name="head_upsample")
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
        #projection = self.summarizer(inputs)
        local_map = self.local_path(inputs,training=training)
        local_map2 = self.local_path2(inputs,training=training) 
        #pos_enc = self.pos_encoding_xy(projection)
        #global_map = self.global_path(projection+pos_enc) # (B,H,W,R)
        #global_map = self.upsample(desc_posc)
        #return self.score_gen(global_map+local_map), global_map #desc_posc #tf.reshape(,shape=(B,-1)) #0.5+tf.clip_by_value(0.58*tf.nn.tanh(0.7*att_map),-0.5,0.5) #tf.nn.sigmoid(att_map)# self.dropout_v(V,training=training),self.dropout_h(H,training=training)# (B,R,K),(B,R,K)  
        return self.score_gen(local_map2+local_map), local_map
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

        self.head_mixer_w = layers.Conv2D(
            filters=1,
            kernel_size=(1,1),
            strides=(1,1),
            padding="same",
            name="Conv2D_scores",
            activation="sigmoid",
            kernel_initializer= "glorot_uniform",#initializers.RandomNormal(mean=0.0, stddev=0.001),
            bias_initializer=tf.constant_initializer(0.0),
            use_bias=True,
            trainable = self.trainable
        )
        
        """
        self.add_weight(
                                    name="hmixerw",
                                    shape=(1,1,1,self.head_num),
                                    initializer="glorot_uniform",
                                    trainable=self.trainable,
                                    constraint = constraints.NonNeg()
                                    #constraint=ConvexComb1D()
                                    )
        """
        
        """
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
        """


        # Spatial Attention heads
        self.spatial_heads = [SpatialEnergyHead(K_size=self.feat_size,
                                                trainable=self.trainable,
                                                name=f"SpatialHead_{u}") 
                                                for u in range(self.head_num)] 
        
        self.global_path = _GlobalPathLayer(name="global_path")

        # Set spatial heads as attributes to ensure full serialization.
        for k,layer in enumerate(self.spatial_heads):
            self.__setattr__(f"sam_{k}", layer)

        # Conv layer for combining the spatial heads' outputs.
        """
        self.score_gen = layers.Conv2D(
            filters=1,
            kernel_size=(1,1),
            strides=(1,1),
            padding="same",
            name="AttConv2D_scores",
            activation=None,
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= "glorot_uniform",#initializers.RandomNormal(mean=0.0, stddev=0.001),
            bias_initializer=tf.constant_initializer(0.0),
            use_bias=True,
            trainable = self.trainable
        )

        #self.softmax_layer = layers.Softmax(axis=[1,2])
        """
        
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
        # Compute the mean energy 2D 
        _inputs = tf.clip_by_value(inputs,-1e3,1e3) #B,H,W,C
        _inputs = tf.reduce_mean(tf.square(_inputs),axis=-1,keepdims=True) #B,H,W,1
        #projection = self.gap_proj(_inputs)
        #_inputs = self.in_ln(inputs)
        B = tf.shape(_inputs)[0] 
        K = self.feat_size # Width or Height
        #spatial_desc = self.summarizer(inputs)
        #print(self.head_mixer_w)
        global_scores = self.global_path(_inputs)
        heads_outs = [self.spatial_heads[u](_inputs,training=training) for u in range(self.head_num)]
        stacked_outs_local = tf.concat([heads_outs[u][0] for u in range(self.head_num)],axis=-1)
        outs_aggregated_local =  tf.reduce_sum(stacked_outs_local,axis=-1,keepdims=True)

        #stacked_outs = tf.concat([heads_outs[u][0] for u in range(self.head_num)],axis=-1)
        #stacked_outs = tf.stack([heads_outs[u][0] for u in range(self.head_num)],axis=-1) #(B,H,W,D,Heads)
        #stacked_outs = tf.concat([heads_outs[u][0] for u in range(self.head_num)],axis=-1) #(B,H,W,Heads)
        
        #stacked_outs_w = self.head_mixer_w*stacked_outs #
        #outs_aggregated = tf.reduce_sum(stacked_outs_w,axis=-1,keepdims=True) #(B,H,W,1) #(B,H,W,D)

        #outs_aggregated =  self.head_mixer_w(stacked_outs)
        

        #stacked_latent = tf.stack([heads_outs[u][1] for ua in range(self.head_num)],axis=-1)
        #scores =  0.1+tf.clip_by_value(0.95*tf.nn.sigmoid((self.score_gen(outs_aggregated)))-0.05,0.0,0.9)
        #scores =  tf.clip_by_value(1.0+0.5*tf.nn.tanh((self.score_gen(outs_aggregated))),0.1,1.5)
        
        #scores =  tf.clip_by_value(tf.nn.sigmoid((self.score_gen(outs_aggregated))),0.01,1.0)
        scores =  tf.clip_by_value(outs_aggregated_local+0.6*global_scores,0.01,1.0)

        # Enforce orthogonality between heads
        head_outs_flatten = tf.reshape(tf.transpose(stacked_outs_local, [0,3,1,2]),shape=(B,self.head_num,-1))# B x H x K^2
        heads_norm = tf.math.l2_normalize(head_outs_flatten,epsilon=1e-6, axis=-1)    # B x H x K x K
        gram = tf.abs(tf.matmul(heads_norm, heads_norm, transpose_b=True))  # (B, H, H)
        eye = tf.eye(tf.shape(gram)[1], batch_shape=[tf.shape(gram)[0]])
        off_diag = gram - eye
        orthogonality_reg = tf.reduce_sum(tf.square(off_diag),axis=[1,2])/tf.reduce_sum(1.0-eye,axis=[1,2])
        attention_loss = (self.feat_size/64.0)*1e-6*tf.reduce_mean(orthogonality_reg)
        self.add_loss(tf.minimum(attention_loss,1e-5))

        # Enforce higher variance at the heads' features
        #latent_mean = tf.reduce_mean(stacked_outs,axis=-1,keepdims=True)
        #latent_var = tf.reduce_mean(tf.square(stacked_outs-latent_mean),axis=-1)
        #var_reg = tf.reduce_mean(1/tf.maximum(latent_var,0.001))
        #self.add_loss((64.0/self.feat_size)*1e-7*var_reg)
        
        return scores #scores_raw # (B,K,K,1)=(B,H,W,1)
    
    def build(self, input_shape):
        super().build(input_shape)
