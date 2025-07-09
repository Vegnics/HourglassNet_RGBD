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
        rank: int = None,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.K_size = K_size
        self.kernel_initializer = kernel_initializer
        self.rank = rank
        # Create layers

        # Shared FC layer
        #self.shared_fc = layers.Dense(
        #        units=self.K_size//2,
        #        activation=None,
        #        use_bias=True,
        #        kernel_initializer='glorot_uniform',
        #        trainable=self.trainable
        #)
        self.bn = layers.BatchNormalization(axis=-1,name="head_bn")

        self.pos_encoding = layers.Embedding(input_dim=self.K_size, output_dim=8,name="pos_encoding")

        self.merger = SequentialLayer(
            [
                layers.Conv2D(
                filters=4,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                name="merger0",
                activation="gelu",
                kernel_regularizer= None,
                kernel_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True),

                layers.MaxPooling2D(pool_size=(2,2),name="merger_mp0"),

                layers.Conv2D(
                filters=self.rank,
                kernel_size=(1,1),
                strides=(1,1),
                padding="same",
                name="merger1",
                activation="gelu",
                kernel_regularizer= None,
                kernel_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
                use_bias=True),

                layers.MaxPooling2D(pool_size=(2,2),name="merger_mp1")
            ],
            name="head_merger",
            trainable=self.trainable
        )


        # FC for the vertical components (row vectors)
        """
        self.Vgen0 = layers.Dense(
                units=self.K_size//2,
                activation= None,
                use_bias=True,
                bias_initializer=initializers.Constant(-1.0),
                kernel_initializer='glorot_uniform',
                kernel_constraint=constraints.MaxNorm(2.1),
                name = "Vgen_FC0",
                trainable=self.trainable,
                )
        """
        self.Vgen1 = layers.Dense(
                units=self.K_size//2,
                activation= None,
                use_bias=True,
                bias_initializer="zeros",
                kernel_initializer='glorot_uniform',
                #kernel_constraint=constraints.MaxNorm(2.1),
                name = "Vgen_FC1",
                trainable=self.trainable,
                )
        
        # FC for the horizontal components (column vectors)
        """
        self.Hgen0 = layers.Dense(self.K_size//2,
                activation=None,
                use_bias=True,
                bias_initializer=initializers.Constant(1.0),
                kernel_initializer='glorot_uniform',
                kernel_constraint=constraints.MaxNorm(2.1),
                name = "Hgen_FC0",
                trainable=self.trainable,
                )
        """ 

        self.Hgen1 = layers.Dense(self.K_size//2,
                activation=None,
                use_bias=True,
                bias_initializer="zeros",
                kernel_initializer='glorot_uniform',
                name = "Hgen_FC1",
                trainable=self.trainable,
                )
        
        # Dropout applied to the vertical comp.
        self.dropout_v = layers.Dropout(0.1,name="dpout_v")
        self.ln_v = layers.LayerNormalization(axis=-1,name="ln_v")

        # Dropout applied to the horizontal comp.
        self.dropout_h = layers.Dropout(0.1,name="dpout_h")
        self.ln_h = layers.LayerNormalization(axis=-1,name="ln_h")
        
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
        lin_xy = tf.range(self.K_size)
        X,Y = tf.meshgrid(lin_xy,lin_xy)
        pos_encx = tf.tile(tf.expand_dims(self.pos_encoding(X),axis=0),(B,1,1,1))
        pos_ency = tf.tile(tf.expand_dims(self.pos_encoding(Y),axis=0),(B,1,1,1))
        _inputs = self.bn(inputs)
        _inputs = tf.concat([_inputs,pos_encx,pos_ency],axis=-1)
        desc = self.merger(_inputs)
        desc = tf.reshape(desc,shape=(B,(self.K_size//4)**2,self.rank))
        desc_t = tf.transpose(desc,perm=[0,2,1]) # (B,R,Kin)
        V = self.ln_v(tf.nn.relu(self.Vgen1(desc_t))) # Vertical components : (B,R,K)
        H = self.ln_h(tf.nn.relu(self.Hgen1(desc_t))) # Horizontal components : (B,R,K)  
        #V = tf.transpose(V,perm=[0,2,1])  # (B,K,R)
        #V0 = tf.nn.relu(self.shared_fc(self.Vgen0(_input)))
        #V = tf.nn.relu(self.Vgen1(V0)) # Vertical components : (B,R,K)
        #H0 = tf.nn.relu(self.shared_fc(self.Hgen0(_input)))
        #H = tf.nn.relu(self.Hgen1(H0)) # Horizontal components : (B,R,K)
        #V = tf.transpose(V,perm=[0,2,1])  # (B,K,R)
        return self.dropout_v(V,training=training),self.dropout_h(H,training=training)# (B,R,K),(B,R,K)  
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
        rank: int = 4,
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
        self.rank = rank
        # Create layers
        self.in_bn = layers.BatchNormalization(axis=-1,trainable=self.trainable)
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
        # Layer normalization for the heads' outputs
        self.ln = layers.LayerNormalization(axis=-1,
                                            trainable=self.trainable,
                                            name="LN_sam")
        """
        self.summarizer = SequentialLayer(
            [   layers.Conv2D( filters=1,
                            kernel_size=(1,1),
                            kernel_initializer="glorot_uniform",
                            kernel_constraint=constraints.NonNeg(),
                            name="summ_conv1",
                            ),

                layers.MaxPooling2D(pool_size=(2,2),
                                    strides=(2,2),
                                    padding="valid",
                                    name="summ_pool",
                                ),

                layers.Conv2D(filters=self.rank,
                                kernel_size=(3,3),
                                strides=(2,2),
                                padding="same",
                                activation="relu",
                                name="summ_conv2"
                                )
            ],
            name="SAM_summarizer"
        )
        """

        self.summarizer = SequentialLayer(
            [   #layers.DepthwiseConv2D(kernel_size=(3,3),
                #                        strides=(1,1),
                #                        depth_multiplier=1,
                #                        padding="same",
                #                        trainable=self.trainable,
                #                        activation=None,
                #                        name="dwise_downsample"),
        
                layers.Conv2D(filters=32,
                            kernel_size=(1,1),
                            kernel_initializer="glorot_uniform",
                            name="summ_conv1",
                            activation= None #"relu"
                            ),

                #layers.MaxPooling2D(pool_size=(2,2),
                #                    strides=(2,2),
                #                    padding="valid",
                #                    name="summ_pool",
                #                ),

                #layers.Conv2D(filters=self.rank,
                #                kernel_size=(1,1),
                #                padding="same",
                #                activation="relu",
                #                name="summ_conv2"
                #                )
            ],
            name="SAM_summarizer"
        )
        # Spatial Attention heads
        self.spatial_heads = [SpatialEnergyHead(K_size=self.feat_size,rank=self.rank,name=f"SpatialHead_{u}") for u in range(self.head_num)] 
        

        # Set spatial heads as attributes to ensure full serialization.
        for k,layer in enumerate(self.spatial_heads):
            self.__setattr__(f"sam_{k}", layer)

        # Conv layer for combining the spatial heads' outputs.
        """
        self.score_gen = layers.Conv2D(
            filters=1,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            name="AttConv2D_scores",
            activation=None,
            kernel_regularizer= RegL2(1e-6) if self.kernel_reg else None,
            kernel_initializer= "glorot_uniform", #initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=tf.constant_initializer(0.0),
            use_bias=True,
        )
        """

        # 2D positional encodings 
        self.pos_encoding_x = layers.Embedding(input_dim=self.feat_size, output_dim=8,name="pos_encoding_x")
        self.pos_encoding_y = layers.Embedding(input_dim=self.feat_size, output_dim=8,name="pos_encoding_y")
        self.score_gen_v = layers.Dense( units=self.kernel_size,
                                            kernel_initializer="glorot_uniform",
                                            trainable=self.trainable)
        self.score_gen_h = layers.Dense( units=self.kernel_size,
                                            kernel_initializer="glorot_uniform",
                                            trainable=self.trainable)
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
        B = tf.shape(inputs)[0] 
        K = tf.shape(inputs)[1] # Width or Height
        C = tf.shape(inputs)[3] # Channels
        Kd = K//2
        #tiled = tf.reshape(_inputs, (-1, 4, K // 4, 4, K // 4,C))
        #tiled = tf.transpose(tiled, perm=[0, 1, 3, 2, 4,5]) # (B,4,4,K//4,K//4,C)
        #tiled_norm = tf.reduce_sum(tf.math.square(tiled),axis=-1)
        #energy_tile = tf.expand_dims(tf.reduce_mean(tiled_norm,axis=[3,4]),axis=-1) #Nx4x4x1
        #energy_tile = tf.reduce_mean(tf.math.square(tiled),axis=[3,4]) #Nx4x4x1
        
        #mean_e = tf.reduce_mean(energy_tile, axis=[1,2], keepdims=True)
        #std_e = tf.math.maximum(tf.math.reduce_std(energy_tile, axis=[1,2], keepdims=True), 1e-3*tf.ones_like(energy_tile))
        #energy_tile = (energy_tile - mean_e) / std_e # patch-wise energy tile normalization
        #_inputs = self.in_bn(inputs,training=training)
        spatial_desc = self.summarizer(inputs)
        #spatial_desc = tf.reshape(spatial_desc,shape=(B,Kd**2,self.rank))
        lin_xy = tf.range(K)
        X,Y = tf.meshgrid(lin_xy,lin_xy)
        pos_encx = tf.tile(tf.expand_dims(self.pos_encoding_x(X),axis=0),(B,1,1,1))
        pos_ency = tf.tile(tf.expand_dims(self.pos_encoding_y(Y),axis=0),(B,1,1,1))
        
        # Simple 2D positional encoding
        #pos_encx = tf.reshape(self.pos_encoding_x( tf.range(K)//(K//4)),(4,)) # 4
        #pos_ency = tf.reshape(self.pos_encoding_y(tf.range(K)),(4,)) # 4
        
        #pos_encx = self.pos_encoding_x(tf.range(Kd**2)//Kd) # 4
        #pos_ency = self.pos_encoding_y(tf.range(Kd**2)%Kd) # 4
        
        #X,Y = tf.meshgrid(pos_encx,pos_ency) # 2 4x4
        #XY = tf.expand_dims(tf.stack([X,Y],axis=-1),axis=0) # 1x4x4x2
        #pos_encoding = tf.tile(XY,(tf.shape(inputs)[0],1,1,1)) # Nx4x4x2

        #spatial_with_pos = spatial_desc+ pos_encx + pos_ency
        
        spatial_with_pos = spatial_desc #tf.concat([spatial_desc,pos_encx,pos_ency],axis=-1)
        #spatial_with_pos = self.desc_merger(spatial_with_pos)
        #spatial_with_pos = tf.reshape(spatial_with_pos,shape=(B,(Kd//2)**2,self.rank))
        # Input the energy-based spatial descriptor (energy tile,pos encoding) to the Spatial Attention heads  
        #energy_descriptor = tf.reshape(tf.concat([energy_tile,pos_encoding],axis=-1),(B,16,3))
        #stacked_outs = tf.stack([self.spatial_heads[u](energy_descriptor,training=training) for u in range(self.head_num)],axis=-1)
        
        #stacked_outs = tf.stack([self.spatial_heads[u](spatial_with_pos,training=training) for u in range(self.head_num)],axis=-1)
        #stacked_outs = self.ln(stacked_outs) # (B,K,K,self.head_num)
        concat_out_v = tf.concat([self.spatial_heads[u](spatial_with_pos,training=training)[0] for u in range(self.head_num)],axis=-1)
        concat_out_h = tf.concat([self.spatial_heads[u](spatial_with_pos,training=training)[1] for u in range(self.head_num)],axis=-1)
        
        # Enforce orthogonality between heads
        #head_outs_flatten = tf.reshape(tf.transpose(stacked_outs, [0,3,1,2]),shape=(B,self.head_num,-1))# B x H x K^2
        #heads_norm = tf.math.l2_normalize(head_outs_flatten,epsilon=1e-6, axis=-1)    # B x H x K x K
        #gram = tf.matmul(heads_norm, heads_norm, transpose_b=True)  # (B, H, H)
        #eye = tf.eye(tf.shape(gram)[1], batch_shape=[tf.shape(gram)[0]])
        #off_diag = gram - eye
        #orthogonality_reg = tf.reduce_mean(tf.square(off_diag))
        #self.add_loss(1e-5 * orthogonality_reg)
        # Raw scores generation 
        #scores_raw = self.score_gen(stacked_outs)

        V_comp = self.score_gen_v(concat_out_v)
        H_comp = self.score_gen_h(concat_out_h)
        V_comp = tf.transpose(V_comp,perm=[0,2,1])
        scores_raw = tf.expand_dims(tf.matmul(V_comp, H_comp),axis=-1)
        scores = 1.0 + 0.9*tf.nn.tanh(scores_raw)
        #scores = tf.clip_by_value(scores_raw,-2.2,2.2)
        #scores = (tf.nn.sigmoid(scores)-tf.nn.sigmoid(-2.2))/(tf.nn.sigmoid(2.2)-tf.nn.sigmoid(-2.2))
        
        # Enforce higher variance at the scores
        #scores_mean = tf.reduce_mean(scores_raw,axis=[1,2],keepdims=True)
        #scores_var = tf.reduce_mean(tf.square(scores_raw-scores_mean),axis=[1,2])
        #var_reg = tf.reduce_mean(1/tf.maximum(scores_var,0.1))
        #self.add_loss(1e-4*var_reg)
        return scores #scores_raw # (B,K,K,1)=(B,H,W,1)
    
    def build(self, input_shape):
        super().build(input_shape)
