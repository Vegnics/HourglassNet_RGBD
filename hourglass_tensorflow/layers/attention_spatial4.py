from hashlib import file_digest
from matplotlib import axis
from numpy import dtype, shape
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

@tf.function
def tf_attention_softargmax(tensor: tf.Tensor, tau: float = 0.01) -> tf.Tensor:
    # tensor: (B, H, W, C)
    b, h, w, c = tf.unstack(tf.shape(tensor))
    # Flatten spatial
    flat = tf.reshape(tensor, [b, h * w, c])  # (B, HW, C)

    # Temperatured softmax with stabilization
    logits = flat / tf.maximum(tau, 1e-6)
    logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)  # (B, HW, C)
    weights = tf.nn.softmax(logits, axis=1)  # (B, HW, C)

    # Normalized grids in [0,1]
    xs = tf.linspace(0.0, 1.0, w)
    ys = tf.linspace(0.0, 1.0, h)
    xg, yg = tf.meshgrid(xs, ys, indexing="xy")  # (H, W)
    xg = tf.reshape(xg, [1, h * w, 1])  # (1, HW, 1)
    yg = tf.reshape(yg, [1, h * w, 1])  # (1, HW, 1)

    x = tf.reduce_sum(weights * xg, axis=1)  # (B, C)
    y = tf.reduce_sum(weights * yg, axis=1)  # (B, C)
    return tf.stack([x, y], axis=-1)  # (B, C, 2)

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
            kernel_initializer= "he_normal",#initializers.RandomNormal(mean=0.0, stddev=0.001),
            bias_initializer=tf.constant_initializer(0.0),
            use_bias=True,
            trainable = self.trainable
        )
        
    def call(self,inputs ,training = False):
        _inputs_0 = self.global_extractor(inputs)
        pos_encoding = self.pos_encoding_xy(_inputs_0)
        _inputs = _inputs_0 + pos_encoding
        skip1 = self.downskip_1(_inputs)
        x = self.core_1(_inputs)
        x = self.core_1_down(x)+skip1
        skip2 = self.downskip_2(x)
        x = self.core_2(x)
        x = self.core_2_down(x) + skip2
        return self.score_gen(tf.concat([self.upsample_deconv(x),_inputs],axis=-1))
    
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
        
        self.pos_encoding_xy = TFPositionalEncoding2D(16)

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
           name="local_extractor",
           trainable = self.trainable
        )

        """
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

                
                #layers.LayerNormalization(axis=-1,name="ln_local",trainable=self.trainable),

           ],
           name="local_extractor2"
        )
        """

        self.maxpool1 = layers.MaxPool2D(pool_size=(2,2),name="mpool_1")
        self.maxpool2 = layers.MaxPool2D(pool_size=(2,2),name="mpool_2")

        self.upsample1 = layers.UpSampling2D(size=(2,2),interpolation="bilinear",name="upsample1")
        self.upsample2 = layers.UpSampling2D(size=(2,2),interpolation="bilinear",name="upsample2")

        self.local_x2 = SequentialLayer(
            [
                layers.Conv2D(
                    filters=32,#8,
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding="same",
                    name="head_local_conv_2",
                    activation=None,
                    kernel_initializer = "glorot_uniform",
                    bias_initializer=tf.constant_initializer(0.0),
                    use_bias=True,
                    trainable = self.trainable),

                SpatialSoftMax(name="head_softmax1"),
                
                layers.Conv2D(
                    filters=16,#8,
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    name="head_local_conv_3",
                    activation="gelu",
                    kernel_initializer = "glorot_uniform",
                    bias_initializer=tf.constant_initializer(0.0),
                    use_bias=True,
                    trainable = self.trainable),
            ],
            name="head_localx2",
            trainable=self.trainable
        )
        
        

        self.local_x4 = SequentialLayer(
            [
                layers.Conv2D(
                            filters=16,#8,
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            name="head_local_conv_4",
                            activation="gelu",
                            kernel_initializer = "glorot_uniform",
                            bias_initializer=tf.constant_initializer(0.0),
                            use_bias=True,
                            trainable = self.trainable)
            ],
            name="head_localx4",
            trainable=self.trainable
        )


        self.score_gen = SequentialLayer(
            [
                layers.Conv2D(
                filters=1,
                kernel_size=(1,1),
                strides=(1,1),
                padding="same",
                name="HeadConv2D_scores",
                activation="sigmoid",
                kernel_initializer= "glorot_normal",#initializers.RandomNormal(mean=0.0, stddev=0.001),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable
            )
            ],
            name="head_score_gen",
            trainable=self.trainable
        )
        
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
        B = tf.shape(inputs)[0]
        local_map = self.local_path(inputs,training=training)
        #local_map2 = self.local_path2(inputs,training=training)
        local_map2 = self.maxpool1(inputs)
        local_map2 = self.local_x2(local_map2)
        local_map3 = self.maxpool2(local_map2)
        pos_enc = self.pos_encoding_xy(local_map3)
        local_map3 = self.local_x4(local_map3)
        local_map2 = local_map2 + self.upsample2(local_map3)
        local_map = local_map + self.upsample1(local_map2)
        
        outs_flatten = tf.reshape(tf.transpose(local_map, [0,3,1,2]),shape=(B,16,-1))# B x H x K^2
        norm = tf.math.l2_normalize(outs_flatten,epsilon=1e-6, axis=-1)    # B x H x K x K
        gram = tf.abs(tf.matmul(norm, norm, transpose_b=True))  # (B, H, H)
        eye = tf.eye(tf.shape(gram)[1], batch_shape=[tf.shape(gram)[0]])
        off_diag = gram - eye
        orthogonality_reg = tf.reduce_sum(tf.square(off_diag),axis=[1,2])/tf.reduce_sum(1.0-eye,axis=[1,2])
        attention_loss = (self.K_size/64.0)*1e-7*tf.reduce_mean(orthogonality_reg)
        #global_map = self.local_path3(local_map2)  
        #pos_enc = self.pos_encoding_xy(projection)
        #global_map = self.global_path(projection+pos_enc) # (B,H,W,R)
        #global_map = self.upsample(desc_posc)
        #return self.score_gen(global_map+local_map), global_map #desc_posc #tf.reshape(,shape=(B,-1)) #0.5+tf.clip_by_value(0.58*tf.nn.tanh(0.7*att_map),-0.5,0.5) #tf.nn.sigmoid(att_map)# self.dropout_v(V,training=training),self.dropout_h(H,training=training)# (B,R,K),(B,R,K)  
        #return self.score_gen(local_map2+local_map), local_map
        return self.score_gen(local_map), tf.minimum(attention_loss,1e-6)
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
        
        #self.global_path = _GlobalPathLayer(name="global_path")

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
            activation="sigmoid",
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= tf.constant_initializer(0.5), #"he_normal",#initializers.RandomNormal(mean=0.0, stddev=0.001),
            bias_initializer=tf.constant_initializer(0.0),
            use_bias=True,
            trainable = self.trainable
        )

        self.weight_gen = layers.Dense(units=self.head_num,
                                       activation="sigmoid",
                                       trainable=self.trainable,
                                       name="weighter")
        """
        self.head_pool = SequentialLayer(
                [layers.AveragePooling2D(pool_size=(2,2),name="avg_p"),
                layers.MaxPool2D(pool_size=(2,2),name="max_p"),
                ],
                name="head_pooler"
        )
        """

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
        # Compute the mean energy 2D 
        #_inputs = tf.clip_by_value(inputs,-1e4,1e4) #B,H,W,C
        #_inputs_square = tf.reduce_mean(tf.square(tf.clip_by_value(inputs,-1e3,1e3) ),axis=-1,keepdims=True) #B,H,W,1
        _inputsmean = tf.reduce_mean(inputs,axis=-1,keepdims=True) #B,H,W,1
        _inputsmax = tf.reduce_max(inputs,axis=-1,keepdims=True) #B,H,W,1
        _inputs = tf.concat([_inputsmean,_inputsmax],axis=-1)
        #projection = self.gap_proj(_inputs)
        #_inputs = self.in_ln(inputs)
        B = tf.shape(_inputs)[0] 
        K = self.feat_size # Width or Height
        #spatial_desc = self.summarizer(inputs)
        #print(self.head_mixer_w)
        
        #global_scores = self.global_path(_inputs)
        heads_outs = [self.spatial_heads[u](_inputs,training=training) for u in range(self.head_num)]
        heads_losses = tf.stack([heads_outs[u][1] for u in range(self.head_num)],axis=-1)
        stacked_outs_local = tf.concat([heads_outs[u][0] for u in range(self.head_num)],axis=-1)
        heads_mean = tf.reduce_mean(stacked_outs_local,axis=-1,keepdims=True)
        heads_var = tf.reduce_mean(tf.square(stacked_outs_local-heads_mean),axis=(1,2))
        #heads_descriptor = self.head_pool(stacked_outs_local)
        #_dsize = tf.cast((((K//4)**2)*self.head_num),dtype=tf.int32)
        #heads_descriptor = tf.reshape(tf.transpose(heads_descriptor, [0,3,1,2]),shape=(B,_dsize))
        
        head_weights = self.weight_gen(heads_var)
        head_weights = head_weights/tf.reduce_sum(head_weights,axis=-1,keepdims=True)
        head_weights = tf.expand_dims(head_weights,axis=1)
        head_weights = tf.expand_dims(head_weights,axis=1)
        #stacked_view = tf.concat([stacked_outs_local,global_scores],axis=-1)

        stacked_view = tf.concat([stacked_outs_local],axis=-1)
        outs_aggregated_local =  tf.clip_by_value(tf.reduce_sum(stacked_outs_local,axis=-1,keepdims=True),0.01,1.0)
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
        #scores =  tf.clip_by_value(outs_aggregated_local+global_scor es,0.01,1.0)

        #scores = self.score_gen(stacked_view)
        scores = tf.clip_by_value(tf.reduce_sum(stacked_outs_local*head_weights,axis=-1,keepdims=True),0.001,1.0)
        # Enforce orthogonality between heads
        #positions = tf_attention_softargmax(stacked_outs_local) # B,2,H
        #positions_mean = tf.reduce_mean(positions,axis=-1,keepdims=True)
        #positions_var = tf.reduce_mean(tf.square(positions-positions_mean),axis=-1)
        #positions_reg = tf.reduce_mean(tf.reduce_sum(positions_var,axis=-1))

        #head_outs_flatten = tf.reshape(tf.transpose(stacked_outs_local, [0,3,1,2]),shape=(B,self.head_num,-1))# B x H x K^2
        #heads_norm = tf.math.l2_normalize(head_outs_flatten,epsilon=1e-6, axis=-1)    # B x H x K x K
        #gram = tf.abs(tf.matmul(heads_norm, heads_norm, transpose_b=True))  # (B, H, H)
        #eye = tf.eye(tf.shape(gram)[1], batch_shape=[tf.shape(gram)[0]])
        #off_diag = gram - eye
        #orthogonality_reg = tf.reduce_sum(tf.square(off_diag),axis=[1,2])/tf.reduce_sum(1.0-eye,axis=[1,2])
        #attention_loss = (self.feat_size/64.0)*1e-7*tf.reduce_mean(orthogonality_reg)
        #self.add_loss(tf.minimum(attention_loss,1e-6))
        self.add_loss(tf.reduce_mean(heads_losses))
        # Enforce higher variance at the heads' features
        #latent_mean = tf.reduce_mean(stacked_outs,axis=-1,keepdims=True)
        #latent_var = tf.reduce_mean(tf.square(stacked_outs-latent_mean),axis=-1)
        #var_reg = tf.reduce_mean(1/tf.maximum(latent_var,0.001))
        #self.add_loss((64.0/self.feat_size)*1e-7*var_reg)
        
        return scores,tf.concat([stacked_view,scores],axis=-1) #stacked_outs_local #scores_raw # (B,K,K,1)=(B,H,W,1)
    
    def build(self, input_shape):
        super().build(input_shape)
