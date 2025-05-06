import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish
from keras.regularizers import L2 as RegL2
from keras.saving import register_keras_serializable
from keras import constraints
from keras import initializers

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
                bias_initializer=initializers.Constant(-1.0),
                kernel_initializer='glorot_uniform',
                kernel_constraint=constraints.MaxNorm(2.1),
                name = "Vgen_FC",
                trainable=self.trainable,
                )
        
        self.Hgen = layers.Dense(self.K_size,
                activation=None, #None,
                use_bias=True,
                bias_initializer=initializers.Constant(1.0),
                kernel_initializer='glorot_uniform',
                name = "Hgen_FC",
                trainable=self.trainable,
                )
        
        self.Hln = layers.LayerNormalization(axis=-1,
                                            trainable=self.trainable,
                                            name="HLN_")
        
        self.Vln = layers.LayerNormalization(axis=-1,
                                            trainable=self.trainable,
                                            name="VLN_")
        
        self.dropout_ = layers.Dropout(0.1,name="dpout_")
        #self.dropout_h = layers.Dropout(0.05,name="dpout_h")
        
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
        V = tf.nn.relu(self.Vgen(_input))
        H = tf.nn.relu(self.Hgen(_input))   
        #V = self.Vln(v)  # (B,3,K)
        #H = self.Hln(h)
        #H = self.dropout_h(tf.math.l2_normalize(h), training=training)  # (B,3,K)
        V = tf.transpose(V,perm=[0,2,1])                # (B,K,3)
        #H = tf.reshape(H, (-1, 2, self.K_size))                # (B,2,K)

        return self.dropout_(tf.matmul(V, H),training=training)                            # (B,K,K)
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
        """
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
        """
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
            kernel_initializer= initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=tf.constant_initializer(0.0),
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
        #_inputs = self.bn(inputs,training=training)
        _inputs = tf.clip_by_value(inputs,-1e4,1e4)
        #projection = self.gap_proj(_inputs)
        B = tf.shape(inputs)[0]
        K = tf.shape(inputs)[1]
        C = tf.shape(inputs)[3]
        #_inputs = tf.reduce_mean(tf.math.square(inputs),keepdims=True,axis=-1)
        tiled = tf.reshape(_inputs, (-1, 4, K // 4, 4, K // 4,C))
        tiled = tf.transpose(tiled, perm=[0, 1, 3, 2, 4,5])
        tgrid = tf.range(0,4,1,dtype=tf.float32)/3.0
        X,Y = tf.meshgrid(tgrid,tgrid)
        XY = tf.expand_dims(tf.stack([X,Y],axis=-1),axis=0)
        tiled_norm = tf.reduce_sum(tf.math.square(tiled),axis=-1)
        energy_tile = tf.expand_dims(tf.reduce_mean(tiled_norm,axis=[3,4]),axis=-1) #Nx4x4x1
        #energy_tile = tf.reduce_mean(tf.math.square(tiled),axis=[3,4]) #Nx4x4x1
        mean_e = tf.reduce_mean(energy_tile, axis=[1,2], keepdims=True)
        std_e = tf.math.maximum(tf.math.reduce_std(energy_tile, axis=[1,2], keepdims=True), 1e-3*tf.ones_like(energy_tile))
        energy_tile = (energy_tile - mean_e) / std_e
        #energy_tile = tf.math.l2_normalize(energy_tile,axis=[1,2],epsilon=1e-3)
        pos_encoding = tf.tile(XY,(tf.shape(inputs)[0],1,1,1)) # Nx4x4x2
        energy_descriptor = tf.reshape(tf.concat([energy_tile,pos_encoding],axis=-1),(B,16,3))
        stacked_outs = tf.stack([self.spatial_heads[u](energy_descriptor,training=training) for u in range(self.head_num)],axis=-1)
        stacked_outs = self.ln(stacked_outs)

        #head_mean = tf.math.reduce_std(stacked_outs, axis=-1)  # B x K x K x self.head_num
        head_mean = tf.reduce_mean(stacked_outs,keepdims=True,axis=-1)
        head_var = tf.reduce_mean(tf.math.square(stacked_outs-head_mean),axis=[1,2,3])+1e-6
        loss_diversity = tf.reduce_mean(tf.math.sqrt(head_var))
        #loss_diversity = tf.exp(-10.0 * tf.clip_by_value(head_var, 0.0, 5.0))
        #loss_diversity = tf.math.exp(-1.0*head_var)  # penalize low diversity
        self.add_loss(-0.00002 * loss_diversity)
        scores_raw = self.score_gen(stacked_outs)
        #scores = tf.nn.sigmoid(self.score_gen(stacked_outs))#tf.clip_by_value(tf.nn.swish(self.score_gen(stacked_outs)),0.0,1.0) 
        return scores_raw
    
    def build(self, input_shape):
        super().build(input_shape)
