from os import name
from turtle import pos
from annotated_types import T
from cv2 import repeat
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
from hourglass_tensorflow.handlers import train
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
class ChannelSpliterLayer(Layer):
    """
    This layer splits the input tensor into two channels.
    """
    def __init__(self, name: str = None,split_idx: int = None) -> None:
        super().__init__(name=name, trainable=False)
        self.split_idx = split_idx

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Split the input tensor into two channels
        return tf.split(inputs, num_or_size_splits=2, axis=-1)[self.split_idx]


@register_keras_serializable(package="lattentionSpatial")    
class _GlobalPathLayer(Layer):
    def __init__(
            self,
            name: str = None,
            trainable: bool = None,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
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
            kernel_initializer= "glorot_normal",#initializers.RandomNormal(mean=0.0, stddev=0.001),
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
        return self.score_gen(self.upsample_deconv(x))
    
@register_keras_serializable(package="lattentionSpatial") 
class AddPosEncodingLayer(Layer):
    """
    This layer adds positional encoding to the input tensor.
    """
    def __init__(self, name: str = None, ndim:int = None) -> None:
        super().__init__(name=name, trainable=False)
        self.pos_encoding_xy = TFPositionalEncoding2D(ndim) 
        self.ndim = ndim 
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        pos_enc = self.pos_encoding_xy(inputs)
        return inputs + pos_enc
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "ndim": self.ndim,
            },
        }
    
@register_keras_serializable(package="lattentionSpatial") 
class GlobalExtractor(Layer):
    def __init__(self,
        K_size: int = 32,
        rank: int = 6,
        name: str = None,
        trainable: bool = True,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        self.K_size = K_size
        self.rank = rank
        self.trainable = trainable
        
        self.Vgen1 = layers.Dense(
                units=self.K_size//2,
                activation= None,
                use_bias=True,
                bias_initializer="zeros",
                kernel_initializer='glorot_uniform',
                name = "Vgen_FC1",
                trainable=self.trainable,
                )
        
        # FC for the horizontal components (column vectors)
        self.Hgen1 = layers.Dense(self.K_size//2,
                activation=None,
                use_bias=True,
                bias_initializer="zeros",
                kernel_initializer='glorot_uniform',
                name = "Hgen_FC1",
                trainable=self.trainable,
                )

        self.pos_encoding_xy = AddPosEncodingLayer(name="add_pos_encoding_3",ndim=16)
        self.downsample = layers.MaxPool2D(pool_size=(2,2),name="mpool_2")
        # Dropout applied to the vertical comp.
        self.dropout_v = layers.Dropout(0.1,name="dpout_v")
        # Dropout applied to the horizontal comp.
        self.dropout_h = layers.Dropout(0.1,name="dpout_h")

        self.upsample_deconv = layers.Conv2DTranspose(name="upsample_deconv",
                                                    kernel_size=(3,3),
                                                    filters= 8,
                                                    strides=(2,2),
                                                    trainable=self.trainable,
                                                    padding="same",)
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "K_size": self.K_size,
                "rank": self.rank,
            },
        }
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        B = tf.shape(inputs)[0] 
        desc = tf.reshape(inputs,shape=(B,(self.K_size//2)**2,self.rank))
        desc = tf.transpose(desc,perm=[0,2,1])
        V = self.Vgen1(desc) # Vertical components : (B,R,K)
        H = self.Hgen1(desc) # Horizontal components : (B,R,K)  
        V = self.dropout_v(V,training=training)
        H = self.dropout_h(H,training=training)
        global_map = tf.expand_dims(tf.nn.gelu(tf.matmul(V,H,transpose_a=True)),axis=-1)
        global_map = tf.reshape(global_map,shape=(B,self.K_size//2,self.K_size//2,1))
        return self.upsample_deconv(global_map) # B,K,K,1
    def build(self, input_shape):
        super().build(input_shape)  
    
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
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.K_size = K_size
        self.kernel_initializer = kernel_initializer
        # Create layers
        # Shared FC layer
        
        self.pos_encoding_xy = TFPositionalEncoding2D(16)


        # Using a linear projection layer (Use NonNeg() constraint ?)
        #"""
        self.in_proj = layers.Conv2D(filters=8,
                                    kernel_size=(1,1),
                                    strides=(1,1),
                                    padding="same",
                                    name="in_proj",
                                    activation=None,
                                    kernel_initializer="glorot_uniform",
                                    bias_initializer=tf.constant_initializer(0.0),
                                    #kernel_constraint=constraints.NonNeg(),
                                    use_bias=True,
                                    trainable=self.trainable)
        #"""

        # Local attetion path at the original resolution
        self.local_path = SequentialLayer(
           [
              layers.Conv2D(
                filters=8,#8,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="head_local_conv_1",
                activation="gelu",
                kernel_initializer = "glorot_uniform",
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable),

                #layers.LayerNormalization(axis=-1,name="ln_local",trainable=self.trainable),
           ],
           name="local_extractor",
           trainable = self.trainable
        )

        """
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
                
                AddPosEncodingLayer(name="add_pos_encoding_1",ndim=16),
        """

        # Local attetion path at half the original resolution
        self.local_x2 = SequentialLayer(
            [
                
                #layers.AveragePooling2D(pool_size=(2,2),name="mpool_1"),

                layers.Conv2D(
                    filters=8,#8,
                    kernel_size=(3,3),
                    strides=(2,2),
                    padding="same",
                    name="head_local_conv_4",
                    activation="gelu",
                    kernel_initializer = "glorot_uniform",
                    bias_initializer=tf.constant_initializer(0.0),
                    use_bias=True,
                    trainable = self.trainable),
                
                #AddPosEncodingLayer(name="add_pos_encoding_2",ndim=8),

                #GlobalExtractor(name="global_extractor", K_size=self.K_size ,rank=8,trainable=self.trainable),

                layers.UpSampling2D(size=(2,2),interpolation="bilinear",name="upsample1"),

                #layers.LayerNormalization(axis=-1,name="ln_local_1",trainable=self.trainable),
            ],
            name="head_localx2",
            trainable=self.trainable
        )



        # Add positional encoding to the concatenated features (local paths)????
        #self.add_pos_encoding = 
        
        """
        self.score_gen = SequentialLayer(
            [
                layers.Conv2D(
                filters=8,
                kernel_size=(1,1),
                strides=(1,1),
                padding="same",
                name="HeadConv2D_scores",
                activation="gelu",
                kernel_initializer= "glorot_uniform",#initializers.RandomNormal(mean=0.0, stddev=0.001),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable
            ),
            #    layers.UpSampling2D(size=(2,2),
            #        interpolation="bilinear",
            #        name="upsample_scores"
            #    )
            ],
            name="head_score_gen",
            trainable=self.trainable
        )
        """

        # Single channel head's attention map
        self.score_gen = SequentialLayer(
            [
                layers.Conv2D(
                filters=1,
                kernel_size=(1,1),
                strides=(1,1),
                padding="same",
                name="HeadConv2D_scores",
                activation=None,
                kernel_initializer= "glorot_uniform",#,#initializers.RandomNormal(mean=0.0, stddev=0.001),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True,
                trainable = self.trainable
                ),
            #    layers.UpSampling2D(size=(2,2),
            #        interpolation="bilinear",
            #        name="upsample_scores"
            #    )
            ],
            name="head_score_gen",
            trainable=self.trainable
        )

        self.out_ln = layers.LayerNormalization(axis=-1,name="ln_out",trainable=self.trainable)

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "kernel_initializer": self.kernel_initializer,
                "K_size": self.K_size,
                "rank": self.rank,
            },
        }

    def call(self, inputs, y_true=None, training=False):
    #def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        #projection = self.summarizer(inputs)
        B = tf.shape(inputs)[0]
        #_inputs = self.linear_proj(inputs) # B,H,W,16
        #_inputsmean = tf.reduce_mean(inputs,axis=-1,keepdims=True) #B,H,W,c
        #_inputsmax = tf.reduce_max(inputs,axis=-1,keepdims=True) #B,H,W,1
        #_inputs = tf.concat([_inputsmean,_inputsmax],axis=-1)
        
        # Layer Normalization before projection ???
        _inputs = self.out_ln(inputs)
        _inputs = self.in_proj(_inputs) # B,H,W,16 (Use NonNeg() constraint ?)
        

        local_map = self.local_path(_inputs,training=training)
        local_map2 = self.local_x2(_inputs,training=training)
        #local_map2 = self.add_pos_encoding(local_map2) # Add positional encoding ???
        local_map_concat = tf.concat([local_map,local_map2],axis=-1) #local_map + local_map2
        

        #local_map_concat = local_map + local_map2 #local_map_concat

        weights = tf.ones((1,16))
        if len(self.score_gen.weights)>0:
            weights0 = tf.reshape(self.local_path.weights[0],shape=(1,-1))
            weights0 = weights0/(tf.reduce_sum(tf.abs(weights0),axis=-1,keepdims=True)+1e-6) 
            weights1 = tf.reshape(self.local_x2.weights[0],shape=(1,-1))
            weights1 = weights1/(tf.reduce_sum(tf.abs(weights1),axis=-1,keepdims=True)+1e-6) 
            weights2 = tf.reshape(self.score_gen.weights[0],shape=(1,-1))
            weights2 = weights2/(tf.reduce_sum(tf.abs(weights2),axis=-1,keepdims=True)+1e-6)
            weights = tf.concat([weights0,weights1],axis=-1)
        
        #local_map_concat = self.out_ln(local_map_concat)
        scores_raw = self.score_gen(local_map_concat)
        return scores_raw, weights #self.score_gen(local_map), weights #self.score_gen.weights[0] # tf.minimum(attention_loss,1e-6)
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
        include_metric: bool = False,
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
        self.include_metric = include_metric
        self.alpha = self.add_weight(shape=[],initializer="zeros",dtype=tf.float32)
        self.tau = self.add_weight(shape=[],initializer="zeros",dtype=tf.float32)
        # Create layers

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
        self.score_gen = SequentialLayer(
            [
                layers.Conv2D(
                filters=1,
                kernel_size=(1,1),
                strides=(1,1),
                padding="same",
                name="AttConv2D_scores",
                activation=None,
                kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
                kernel_constraint=ConvexComb(),
                kernel_initializer= "glorot_uniform",
                #bias_initializer=tf.constant_initializer(0.0001),
                use_bias= False, #False,
                trainable = self.trainable
            ),
            ],
            name = "score_aggregation",
            trainable = self.trainable
        )
        
        #self.attention_reg_metric = keras.metrics.Mean(name=f"Reg_{name[5:]}") if self.include_metric else None

        self.ln = layers.LayerNormalization(axis=-1,name="ln_spatial",trainable=self.trainable)
        
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

    def call(self, inputs, y_true=None, training=False):
    #def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        # Compute a rich collapsed feature
        #S_mean = self.S_mean(inputs,training=training) # B,H,W,8
        #S_max = self.S_max(inputs,training=training) # B,H,W,8
        #S = tf.concat([S_mean,S_max],axis=-1) # B,H,W,16
        #S = self.ln(S) # B,H,W,16
        #_inputsmean = tf.reduce_mean(inputs,axis=-1,keepdims=True) #B,H,W,1
        #_inputsmax = tf.reduce_max(inputs,axis=-1,keepdims=True) #B,H,W,1
        #_inputs = tf.concat([_inputsmean,_inputsmax],axis=-1)
        #tf.print(self.name)
        B = tf.shape(inputs)[0]
        C = tf.shape(inputs)[-1] 
        K = self.feat_size # Width or Height
        splitted = tf.split(inputs, num_or_size_splits=self.head_num, axis=-1) # B,H,W,filters/self.head_num

        # Compute the local descriptors from the spatial heads
        #heads_outs = [self.spatial_heads[u](_inputs,training=training) for u in range(self.head_num)]
        heads_outs = [self.spatial_heads[u](splitted[u],training=training) for u in range(self.head_num)]
        
        # Not used for now
        heads_kernels = tf.stack([tf.squeeze(heads_outs[u][1]) for u in range(self.head_num)],axis=-1)
        heads_losses = tf.stack([heads_outs[u][1] for u in range(self.head_num)],axis=-1)
        
        # Used to compute the final attention scores
        stacked_outs_local = tf.concat([heads_outs[u][0] for u in range(self.head_num)],axis=-1)# B,H,W,C*Heads
        
        stacked_outs = tf.stack([heads_outs[u][0] for u in range(self.head_num)],axis=-1) # B,H,W,C,Heads
        stacked_out_mean = tf.reduce_mean(stacked_outs,axis=-1) # B,H,W,C
        #stacked_view = tf.concat([stacked_outs_local],axis=-1)
        


        # NOT USED FOR NOW
        # Enforce orthogonality between heads (orthogonality regularization)
        stacked_ = tf.transpose(stacked_outs,perm=[0,3,4,1,2]) # B x C x Heads x H x W
        stacked_ = tf.reduce_mean(stacked_,axis=1) # B,Heads,H,W
        stacked_view = tf.transpose(stacked_,perm=[0,2,3,1]) # B,H,W,Heads
        #heads_flatten = tf.reshape(stacked_,shape=(B,self.head_num,-1)) # 
        heads_norm = tf.math.l2_normalize(heads_kernels,epsilon=1e-6, axis=0) # F,Heads   # B x H x K x K
        #heads_norm = tf.math.l2_normalize(heads_flatten,epsilon=1e-6, axis=-1)
        gram = tf.abs(tf.matmul(heads_norm, heads_norm, transpose_a=True))  # (H, H)
        eye = tf.eye(tf.shape(gram)[1])
        off_diag = gram - eye
        orthogonality_reg = tf.reduce_sum(tf.square(off_diag),axis=[0,1])/tf.reduce_sum(1.0-eye,axis=[0,1])
        #eye = tf.eye(tf.shape(gram)[1], batch_shape=[tf.shape(gram)[0]])
        #off_diag = gram - eye
        #orthogonality_reg = tf.reduce_sum(tf.square(off_diag),axis=[1,2])/tf.reduce_sum(1.0-eye,axis=[1,2])

        
        
        #if self.include_metric:
        #    metric = tf.reduce_mean(orthogonality_reg) #rec_penalty #
        #    self.attention_reg_metric.update_state(metric)
        attention_loss = (self.feat_size/64.0)*1e-3*tf.reduce_mean(orthogonality_reg)
        #self.add_loss(attention_loss)
        #self.add_loss(tf.minimum(attention_loss,1e-4))
        #self.add_loss(tf.reduce_mean(heads_losses))
        

        #group_c = self.filters//self.head_num
        #scores_l = []
        #for u in range(self.head_num):
        #    att_map = tf.expand_dims(stacked_outs_local[:,:,:,u],axis=-1) # B,H,W,1
        #    scores_l.append(att_map*tf.ones(shape=(1,1,1,group_c)))
        #scores = tf.concat(scores_l,axis=-1) # B,H,W,filters
        
        stacked_view = tf.nn.sigmoid(1.5*stacked_view)
        #scores = tf.nn.sigmoid(scores/2.0)
        
        #scores = tf.nn.sigmoid(self.score_gen(stacked_outs_local)/2.0)

        # Use linear (No Convex Comb) for combination and then sigmoid
        scores = self.score_gen(tf.nn.sigmoid(1.5*stacked_outs_local))
        #return tf.nn.sigmoid(scores/2.0),tf.concat([stacked_view,scores],axis=-1) #stacked_outs_local #scores_raw # (B,K,K,1)=(B,H,W,1)
        return scores,stacked_view

    def reset_metrics(self):
        self.attention_reg_metric.reset_states()

    def build(self, input_shape):
        super().build(input_shape)#
