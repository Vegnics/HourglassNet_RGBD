from xml.etree.ElementInclude import include
from numpy import stack
import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.saving import register_keras_serializable


from hourglass_tensorflow.layers.residual_lora import ResidualLayer,ResidualLayerIn
from hourglass_tensorflow.layers.dummy_layers import zeroLayer,IdentityLayer
from hourglass_tensorflow.layers.sequential_layer import SequentialLayer

from hourglass_tensorflow.layers.linear_projection import LinearProjectionLoRA as LinearProjection

#Experimenting with LoRA at the linear projection layers
#from hourglass_tensorflow.layers.linear_projection import LinearProjectionLoRA as LinearProjection
#from hourglass_tensorflow.layers.batch_norm_conv_1 import BatchNormConv1Layer


@register_keras_serializable(package="lHourglass")
def generate_residual_layer(layer_type: str ,
                            feature_filters: int ,
                            nblocks: int,
                            name: str = None ,
                            trainable = True,
                            activate_lora = None,
                            include_metric = False,
                            feat_size = None):
        return  ResidualLayer(
                output_filters= feature_filters,
                nblocks = nblocks,
                name=name,
                trainable=trainable,
                use_last_relu=False,
                attentionType=layer_type,
                feat_size = feat_size,
                include_metric = include_metric,
                activate_lora = activate_lora)

@register_keras_serializable(package="lHourglass")
class ResidualWithBNRC(Layer):
    def __init__(
        self,
        output_filters: int = 256,# Number of feature maps
        nblocks: int = None,
        kernel_initializer: str = "glorot_uniform",
        momentum: float = 0.9,
        epsilon: float = 0.0001,
        name: str = None,
        trainable: bool = True,
        attention: str = None,
        feat_size: int = None,
        activate_lora: bool = None,
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store Config
        self.feature_filters = output_filters
        self.attention = attention
        self.nblocks = nblocks
        self.kernel_initializer = kernel_initializer
        self.momentum = momentum
        self.epsilon = epsilon
        self.feat_size = feat_size
        self.activate_lora = activate_lora

        self.conv = layers.Conv2D(
            filters=self.feature_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name="Conv2D",
            activation=None,
            kernel_initializer=self.kernel_initializer,
        )

        self.batch_norm = layers.BatchNormalization(
            axis=-1,
            momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable,
            name="BatchNorm",
        ) 

        self.relu = layers.ReLU(
            name="ReLU",
        )

        self.residual1 =  generate_residual_layer(layer_type=self.attention,
                                                feature_filters=self.feature_filters,
                                                nblocks = self.nblocks,
                                                name="Residual",
                                                trainable=self.trainable,
                                                feat_size=self.feat_size,
                                                include_metric = True,
                                                activate_lora=self.activate_lora)
        
        # LoRA path with rank 4
        #"""
        self.lora = SequentialLayer(
            [       layers.LayerNormalization(axis=-1,
                                              epsilon=1e-4,
                                              name = "lora_ln"),
                    layers.Conv2D(
                        filters=4,
                        kernel_size=1,
                        name="lora_a",
                        activation="gelu",
                        #use_bias=False,
                        kernel_initializer="glorot_uniform",
                    ),

                    layers.Conv2D(
                        filters=self.feature_filters,
                        kernel_size=1,
                        name="lora_b",
                        activation=None,
                        #use_bias=False,
                        kernel_initializer="zeros",
                    )
                ],
                name="lora_path",
                trainable=self.trainable
                ) if self.activate_lora else zeroLayer(self.feature_filters,name="lora_path")
        #"""


    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.feature_filters,
                "nblocks": self.nblocks,
                "attention": self.attention,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "kernel_initializer": self.kernel_initializer,
                "feat_size": self.feat_size,
                "activate_lora": self.activate_lora
            },
        }
    
    # Call method with direct weight manipulation using LoRA
    """
    def call(self,inputs, training=True):
        x = self.residual1(inputs, training=training)
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        conv_kernel = self.conv.kernel
        if self.activate_lora and self.built and self.trainable:
            lora_k = tf.matmul(self.lora_a, self.lora_b) # shape=(kernel_size*kernel_size, filters, filters)
            lora_k = tf.reshape(lora_k, shape=(1, 1, self.in_channels, self.feature_filters))
            conv_kernel = tf.add(conv_kernel, lora_k) # add LoRA weights to the convolution kernel
        #lora = self.lora(x, training=training)
        #x = self.conv(x)
        y = tf.nn.conv2d(x, conv_kernel, strides=[1, 1, 1, 1], padding="SAME")
        return y #+ lora #self.lora_path(inputs)
    """

    def call(self,inputs, training=True):
        x = self.residual1(inputs, training=training)
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        lora = self.lora(x, training=training)
        y = self.conv(x)
        return y #+ lora 
    
    # Build method with direct weight manipulation using LoRA
    """
    def build(self, input_shape):
        self.conv.build(input_shape)
        if self.activate_lora and self.trainable:
            if self.activate_lora and self.trainable:
                self.in_channels = int(input_shape[-1])
                #self.lora.trainable = False
                flattened_kernel_dim = self.in_channels
                self.lora_a = self.add_weight(
                                    shape=(flattened_kernel_dim, 4),
                                    initializer="glorot_uniform",
                                    trainable=True,
                                    name="lora_a_kernel"
                                )
                
                self.lora_b = self.lora_B = self.add_weight(
                                shape=(4, self.feature_filters),
                                initializer="zeros",
                                trainable=True,
                                name="lora_b_kernel"
                            )
            self.batch_norm.momentum = 0.85
            self.batch_norm.trainable = True
            self.conv.trainable = False
            #self.lora.trainable = False
        elif not self.activate_lora and self.trainable:
            #self.lora.trainable = False
            self.conv.trainable = True
            self.batch_norm.trainable = True
        else:
            self.batch_norm.trainable = False
            self.conv.trainable = False
            #self.lora.trainable = False
        super().build(input_shape)
    """

    # Build method with adapter
    def build(self, input_shape):
        if self.activate_lora and self.trainable:
            self.batch_norm.trainable = False
            self.conv.trainable = False
            self.lora.trainable = True
        elif not self.activate_lora and self.trainable:
            self.lora.trainable = False
            self.conv.trainable = True
            self.batch_norm.trainable = True
        else:
            self.batch_norm.trainable = False
            self.conv.trainable = False
            self.lora.trainable = False
        super().build(input_shape)
    
    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """

@register_keras_serializable(package="lHourglasslora") 
class HourglassLayerLora(Layer):
    def __init__(
        self,
        feature_filters: int = 256,# Number of feature maps
        joint_filters_1J: int = 16,  # Number of 1-joint heatmaps
        joint_filters_2J: int = 16,  # Number of 2-joint heatmaps
        downsamplings: int = 4,    # Number of Downsamplings and upsamplings. 
        name: str = None,
        trainable: bool = True,
        intermed: bool = False,
        use_2jointHM: bool = False,
        skip_attention: str = None,
        s2f_attention: str = None,
        f2s_attention: str = None,
        use_kernel_regularization: bool = False,
        residual_nblocks: int = None,
        freeze_attention: bool = False,
        activate_lora: bool = None,
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        self.downsamplings = downsamplings
        self.feature_filters = feature_filters
        self.joint_filters_1J= joint_filters_1J
        self.joint_filters_2J= joint_filters_2J
        self.intermed = intermed
        self.skip_att = skip_attention
        self.s2f_att = s2f_attention
        self.f2s_att = f2s_attention
        self.use_kernel_reg = use_kernel_regularization
        self.use_2jointHM = use_2jointHM
        self.freeze_attention = freeze_attention
        self.residual_nblocks = residual_nblocks
        self.activate_lora = activate_lora
        self.scores_agg = None

        # Used to capture the heads' attention results for visualization
        self.capture = None
        
        # Create Layers
        self.hm1_output = LinearProjection(
            # Layer for heatmaps output.
            filters=self.joint_filters_1J, #output_filters
            kernel_size=1,
            name="HeatmapOutput",
            trainable=trainable,
            activate_lora = self.activate_lora,
            outmax=None,
        )

        self.hm2_output = LinearProjection(
            # Layer for heatmaps output.
            filters=self.joint_filters_2J,#14,
            kernel_size=1,
            name="Heatmap2Output",
            trainable=self.trainable,
            outmax=None,
        ) if self.use_2jointHM else zeroLayer(self.joint_filters_2J,name="Heatmap2Output")

        self.features_hm2 = LinearProjection(
            # Layer for projecting the 2J heatmaps into the feature space.
            filters=self.feature_filters,
            kernel_size=1,
            name="Heatmap2Features",
            trainable=self.trainable,
            outmax=None,
        ) if self.use_2jointHM else zeroLayer(self.feature_filters,name="Heatmap2Features")

        self.residual_2j = ResidualLayer(output_filters=self.feature_filters,
                                           nblocks=self.residual_nblocks,
                                            name="Transit_Output",
                                            trainable=trainable,
                                            epsilon=0.001,
                                            momentum=0.97,
                                            attentionType="NoAM",
                                            activate_lora = self.activate_lora,
        ) if self.use_2jointHM else zeroLayer(self.feature_filters,name="Transit_Output")
        

        self.merge_feats_main = LinearProjection(filters=self.feature_filters,
                                            kernel_size=1,
                                            name="Merge_Feats_main",
                                            trainable=trainable,
                                            activate_lora= self.activate_lora,
        )

        self.merge_feats_1j = LinearProjection(filters=self.feature_filters,
                                            kernel_size=1,
                                            name="Merge_Feats_1J",
                                            trainable=trainable,
                                            activate_lora = self.activate_lora,
        )

        #"""
        self.bn_feats_1j = layers.BatchNormalization(
            axis=-1,
            momentum=0.989,
            epsilon=0.0001,
            trainable=trainable,
            name="BN_Feats_1J",
        )
        #"""

        self.residual_brc = ResidualWithBNRC(
                    output_filters=self.feature_filters,
                    nblocks=self.residual_nblocks,
                    name=f"ResidualWithBNRC",
                    trainable=trainable,
                    attention= self.f2s_att, #"NoAM",#, #"NoAM"
                    feat_size = 64,
                    activate_lora = self.activate_lora,
                )
        
        self.layer_list = {}
        for i in range(self.downsamplings):
            _downsampl = {}
            _downsampl["up_1"] = generate_residual_layer(layer_type= self.skip_att if i!=0 else "NoAM",
                                                           feature_filters=self.feature_filters,
                                                           nblocks=self.residual_nblocks,
                                                           name=f"Step{i}_ResidualUp1",
                                                           trainable=trainable,
                                                           feat_size = 2**(i+3),
                                                           activate_lora = self.activate_lora,) 
            self.__setattr__(f"dstep_{i}_up_1", _downsampl["up_1"])

            _downsampl["low_"] = layers.MaxPool2D(
                pool_size=(2, 2),
                padding="valid",
                name=f"Step{i}_MaxPool",
                trainable=trainable,
            )
            self.__setattr__(f"dstep_{i}_low_", _downsampl["low_"])

            _downsampl["low_1"] = generate_residual_layer(layer_type= self.s2f_att if i!=0 else "NoAM",
                                                            feature_filters=self.feature_filters,
                                                            nblocks=self.residual_nblocks,
                                                            name=f"Step{i}_ResidualLow1",
                                                            trainable=trainable,
                                                           feat_size = 2**(i+2),
                                                           activate_lora = self.activate_lora,)
            self.__setattr__(f"dstep_{i}_low_1", _downsampl["low_1"])

            if i == 0:
                _downsampl["low_2"] = generate_residual_layer(layer_type= "NoAM",
                                                                feature_filters=self.feature_filters,
                                                                nblocks=self.residual_nblocks,
                                                                name=f"Step{i}_ResidualLow2",
                                                                trainable=trainable,
                                                                feat_size = 2**(i+2),
                                                                activate_lora = self.activate_lora,)
                self.__setattr__(f"dstep_{i}_low_2", _downsampl["low_2"])
            """
            elif i == 3:
                downsampling["low_in"] =  generate_residual_layer(layer_type=self.s2f_att,
                                                                  feature_filters=self.feature_filters,
                                                                  name=f"Step{i}_ResidualMainIn",
                                                                  trainable=self.trainable,
                                                                  kernel_reg=self.use_kernel_reg,
                                                           freeze_attention=self.freeze_attention)
                
                downsampling["low_out"] = generate_residual_layer(layer_type=self.f2s_att,
                                                            feature_filters=self.feature_filters,
                                                            name=f"Step{i}_ResidualMainOut",
                                                            trainable=self.trainable,
                                                            kernel_reg=self.use_kernel_reg,
                                                           freeze_attention=self.freeze_attention)
            """
            _downsampl["low_3"] = generate_residual_layer(layer_type= self.f2s_att if i>0 else "NoAM",
                                                            feature_filters=self.feature_filters,
                                                            nblocks=self.residual_nblocks,
                                                            name=f"Step{i}_ResidualLow3",
                                                            trainable=trainable,
                                                           feat_size = 2**(i+2),
                                                           activate_lora = self.activate_lora,)
            self.__setattr__(f"dstep_{i}_low_3", _downsampl["low_3"])

            _downsampl["up_2"] = layers.UpSampling2D(
                size=(2, 2),
                data_format=None,
                interpolation= "nearest", 
                name=f"Step{i}_UpSampling2D",
                trainable=trainable,
            )
            self.__setattr__(f"dstep_{i}_up_2", _downsampl["up_2"])

            _downsampl["out"] = layers.Add(
                name=f"Step{i}_Add",
                trainable=trainable,
            )
            self.__setattr__(f"dstep_{i}_out", _downsampl["out"])

            self.layer_list[f"STEP_{i}"] = dict(_downsampl)
        ndowns = self.downsamplings
        self.score_ups = [layers.UpSampling2D((2**(ndowns-i),2**(ndowns-i)),
                                             interpolation="nearest",
                                               name=f"Score_Upsampling_{i}") for i in range(self.downsamplings)]
        # endregion
    
    
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "downsamplings": self.downsamplings,
                "feature_filters": self.feature_filters,
                "joint_filters_1J": self.joint_filters_1J,
                "joint_filters_2J": self.joint_filters_2J,
                "intermed": self.intermed,
                "skip_attention": self.skip_att,
                "s2f_attention": self.s2f_att,
                "f2s_attention": self.f2s_att,
                "use_kernel_regularization": self.use_kernel_reg,
                "freeze_attention":self.freeze_attention,
                "use_2jointHM": self.use_2jointHM,
                "residual_nblocks":self.residual_nblocks,
                "activate_lora":self.activate_lora
            },
        }
    
    def _recursive_call(self, input_tensor, y_true, step, training=True):
        step_layers = self.layer_list[f"STEP_{step}"]#self.layer_list[step]
        _input = input_tensor
        #_input = step_layers["low_in"](_input, training=training) if step == 3 else input_tensor
        #if step == 3:
        #    _input = step_layers["low_in"](_input, training=training)
        up_1 = step_layers["up_1"](_input, y_true ,training=training) #Skip        
        low_ = step_layers["low_"](_input, training=training) #MaxPool
        low_1 = step_layers["low_1"](low_, y_true , training=training) #S2F
        
        if step == 0:
            low_2 = step_layers["low_2"](low_1, y_true, training=training) # Dont apply any attention mechanism
        else:
            low_2 = self._recursive_call(low_1, y_true, step=(step - 1), training=training)
        low_3 = step_layers["low_3"](low_2, y_true, training=training) # F2S
        
        if step == 3:
            self.capture = 1.0*step_layers["low_3"].scores_heads
        
        interm = tf.reduce_sum(step_layers["low_3"].scores_heads,axis=[3,4]) #Sum along the heads
        self.scores_agg.append(self.score_ups[step](tf.expand_dims(interm,axis=-1))[:,:,:,0]) # N H W 1 -> N H' W' 1 -> N H' W'

        up_2  = step_layers["up_2"](low_3, training=training) # Upsampling
        out = step_layers["out"]([up_1, up_2], training=training) # Add  
        #if step == 3:
        #    out = step_layers["low_out"](out,training=training)
        return out

    def call(self, inputs, y_true=None, training=False):
        self.scores_agg = []
        #tf.print(tf.shape(y_true))
    #def call(self, inputs, training=False):
        _x = 0.0
        #_x += self._recursive_call(
        #    input_tensor=inputs, step=self.downsamplings - 1, training=training
        #)

        _x += self._recursive_call(
            input_tensor=inputs, y_true=y_true , step=self.downsamplings - 1, training=training
        )

        _x = self.residual_brc(_x,training=training) # Output of the Hourglass module

        stacked_scores = tf.stack(self.scores_agg,axis=-1) # N S H W C
        #tf.print(tf.shape(stacked_scores))
        #self.capture = 1.0*s
        main_feats = self.merge_feats_main(_x)
        intermediate_2jhms = self.hm2_output(_x,training=training) # Linear projection to 2-joint heatmaps
        features_2jhms = self.features_hm2(intermediate_2jhms) # Linear projection back to feature space
        transit_2jhms = self.residual_2j(features_2jhms, training=training) # Residual block for 2-joint heatmaps features
        intermediate_1jhms = self.hm1_output(tf.add_n([_x,transit_2jhms]), training=training) # Linear projection to 1-joint heatmaps
        feats1j = self.merge_feats_1j(intermediate_1jhms) # Linear projection back to the feature space


        # One version with BN and one without it
        feats1j = self.bn_feats_1j(feats1j,training=training) # BN to the 1-joint heatmaps features

        out_tensor = tf.add_n(
            [inputs, main_feats, feats1j],
            name=f"{self.name}_OutputAdd",
        )
        #return self.relu(out_tensor), intermediate#tf.cast(tf.clip_by_value(tf.math.floor(intermediate),0.0,32767.0),dtype=tf.int16)
        return out_tensor,tf.concat([intermediate_1jhms,intermediate_2jhms,stacked_scores],axis=-1),self.capture
    def build(self, input_shape):
        super().build(input_shape) 

    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """