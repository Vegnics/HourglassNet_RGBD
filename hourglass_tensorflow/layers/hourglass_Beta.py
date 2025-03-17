import tensorflow as tf
from keras import layers
from keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

from hourglass_tensorflow.layers.residual import ResidualLayer,ResidualLayerIn
from hourglass_tensorflow.layers.dummy_layers import zeroLayer
#from hourglass_tensorflow.layers.residual_input import ResidualLayerIn
#from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer
#from hourglass_tensorflow.layers.hm_output import HMOut
from hourglass_tensorflow.layers.linear_projection import LinearProjection
#from hourglass_tensorflow.layers.batch_norm_conv_1 import BatchNormConv1Layer
from hourglass_tensorflow.layers.residual_with_attention import ResidualLayerAttention
from hourglass_tensorflow.layers.residual_with_attention_spatial import ResidualLayerAttentionSpatial

def generate_residual_layer(layer_type: str ,
                            feature_filters: int ,
                            nblocks: int,
                            name: str = None ,
                            trainable = True,
                            kernel_reg = False,
                            freeze_attention=False):
        #print("Feature filters",feature_filters)
        if layer_type == "NoAM":
            return  ResidualLayer(
                output_filters= feature_filters,
                nblocks = nblocks,
                name=name,
                trainable=trainable,)
        elif layer_type == "SAM":
            return ResidualLayerAttentionSpatial(
                output_filters=feature_filters,
                nblocks = nblocks,
                name=name,
                trainable=trainable,
                kernel_reg=kernel_reg,
                freeze_attention=freeze_attention,)
        elif layer_type == "FAM":
            return ResidualLayerAttention(
                output_filters=feature_filters,
                nblocks = nblocks,
                name=name,
                trainable=trainable,
                kernel_reg = kernel_reg,
                freeze_attention=freeze_attention)
        else:
            raise Exception(f"The residual layer type: {layer_type} is invalid.")



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
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store Config
        self.feature_filters = output_filters
        self.attention = attention
        self.trainable = trainable
        self.nblocks = nblocks
        self.kernel_initializer = kernel_initializer
        self.momentum = momentum
        self.epsilon = epsilon
        self.trainable = trainable

        self.conv = None
        self.batch_norm = None
        self.relu = None
        self.residual1 = None
    
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.feature_filters,
                "nblocks": self.nblocks,
                "attention": self.attention,
                "trainable": self.trainable,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "kernel_initializer": self.kernel_initializer
            },
        }
    
    def call(self,inputs, training=True):
        x = self.residual1(inputs, training=training)
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        x = self.conv(x)
        return x
    
    def build(self, input_shape):
        print(f"[DEBUG]: {self.name} -- input shape : {input_shape}")
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
            trainable=self.trainable,
            name="BatchNorm",
        ) 

        self.relu = layers.ReLU(
            name="ReLU",
        )
        self.residual1 =  generate_residual_layer(layer_type=self.attention,
                                                feature_filters=self.feature_filters,
                                                nblocks = self.nblocks,
                                                name="Residual",
                                                trainable=self.trainable,)
        super().build(input_shape)
    
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        instance.conv = None
        instance.batch_norm = None
        instance.relu = None
        instance.residual1 = None
        return instance

@register_keras_serializable(package="lHourglass") 
class HourglassLayer(Layer):
    def __init__(
        self,
        feature_filters: int = 256,# Number of feature maps
        in_feature_size: int = 64,
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
        freeze_attention: bool = False
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        self.downsamplings = downsamplings
        self.feature_filters = feature_filters
        self.joint_filters_1J= joint_filters_1J
        self.joint_filters_2J= joint_filters_2J
        self.intermed = intermed
        self.trainable = trainable
        self.skip_att = skip_attention
        self.s2f_att = s2f_attention
        self.f2s_att = f2s_attention
        self.use_kernel_reg = use_kernel_regularization
        self.use_2jointHM = use_2jointHM
        self.freeze_attention = freeze_attention
        self.residual_nblocks = residual_nblocks
        self.in_feature_size = in_feature_size
        # Store Config
        #print("Hourglass filters",self.feature_filters)
        # Init parameters
        self.layer_list = [{} for i in range(self.downsamplings)]
        self.hm1_output = None
        self.hm2_output = None
        self.features_hm2 = None
        self.residual_2j = None
        self.merge_feats_main = None
        self.merge_feats_1j = None
        self.residual_brc = None
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
                "in_feature_size":self.in_feature_size
            },
        }
        
    def _recursive_call(self, input_tensor, step, training=True):
        step_layers = self.layer_list[step]
        _input = input_tensor
        #_input = step_layers["low_in"](_input, training=training) if step == 3 else input_tensor
        #if step == 3:
        #    _input = step_layers["low_in"](_input, training=training)
        up_1 = step_layers["up_1"](_input, training=training) #Skip
        low_ = step_layers["low_"](_input, training=training) #MaxPool
        low_1 = step_layers["low_1"](low_, training=training) #S2F
        if step == 0:
            low_2 = step_layers["low_2"](low_1, training=training) # Dont apply any attention mechanism
        else:
            low_2 = self._recursive_call(low_1, step=(step - 1), training=training)
        low_3 = step_layers["low_3"](low_2, training=training) # F2S
        up_2 = step_layers["up_2"](low_3, training=training) # Upsampling
        out = step_layers["out"]([up_1, up_2], training=training) # Add  
        #if step == 3:
        #    out = step_layers["low_out"](out,training=training)
        return out

    def call(self, inputs, training=False):
        _x = self._recursive_call(
            input_tensor=inputs, step=self.downsamplings - 1, training=training
        )
        _x = self.residual_brc(_x)
        main_feats = self.merge_feats_main(_x)
        intermediate_2jhms = self.hm2_output(_x,training=training)
        features_2jhms = self.features_hm2(intermediate_2jhms) 
        transit_2jhms = self.residual_2j(features_2jhms, training=training)

        intermediate_1jhms = self.hm1_output(tf.add_n([_x,transit_2jhms]), training=training)
        #bpart_feats = self.body_part_residual(intermediate_2jhms,training=training)
        #intermediate_1jhms = self._hm_output(tf.add_n([_x,bpart_feats]), training=training) # Intermediate Heatmap outputs >>>> IMPORTANT
        #intermediate_1jhms = self._hm_output(tf.add_n([_x,bpart_feats]), training=training)

        #_out = self._last_residual(_x,training=training)
        out_tensor = tf.add_n(
            [inputs, main_feats, self.merge_feats_1j(intermediate_1jhms)], #_out
            name=f"{self.name}_OutputAdd",
        )
        #return self.relu(out_tensor), intermediate#tf.cast(tf.clip_by_value(tf.math.floor(intermediate),0.0,32767.0),dtype=tf.int16)
        
        return out_tensor, tf.concat([intermediate_1jhms,intermediate_2jhms],axis=-1)
    def build(self, input_shape):
        print(f"[DEBUG]: {self.name} -- input shape : {input_shape}")
        # Create Layers
        #ConvBatchNormReluLayer
        self.hm1_output = LinearProjection(
            # Layer for heatmaps output.
            filters=self.joint_filters_1J, #output_filters
            kernel_size=1,
            name="HeatmapOutput",
            trainable=self.trainable,
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
                                            trainable=self.trainable,
                                            epsilon=0.001,
                                            momentum=0.97,
        ) if self.use_2jointHM else zeroLayer(self.feature_filters,name="Transit_Output")
        
        self.merge_feats_main = LinearProjection(filters=self.feature_filters,
                                            kernel_size=1,
                                            name="Merge_Feats_main",
                                            trainable=self.trainable,
        )

        self.merge_feats_1j = LinearProjection(filters=self.feature_filters,
                                            kernel_size=1,
                                            name="Merge_Feats_1J",
                                            trainable=self.trainable,
        )

        self.residual_brc = ResidualWithBNRC(
                    output_filters=self.feature_filters,
                    nblocks=self.residual_nblocks,
                    name=f"ResidualWithBNRC",
                    trainable=self.trainable,
                    attention= "NoAM"
                )
        
        for i, downsampling in enumerate(self.layer_list):
            downsampling["up_1"] = generate_residual_layer(layer_type=self.skip_att,
                                                           feature_filters=self.feature_filters,
                                                           nblocks=self.residual_nblocks,
                                                           name=f"Step{i}_ResidualUp1",
                                                           trainable=self.trainable,
                                                           kernel_reg=self.use_kernel_reg,
                                                           freeze_attention=self.freeze_attention)
            downsampling["low_"] = layers.MaxPool2D(
                pool_size=(2, 2),
                padding="valid",
                name=f"Step{i}_MaxPool",
                trainable=self.trainable,
            )
            downsampling["low_1"] = generate_residual_layer(layer_type=self.s2f_att,
                                                            feature_filters=self.feature_filters,
                                                            nblocks=self.residual_nblocks,
                                                            name=f"Step{i}_ResidualLow1",
                                                            trainable=self.trainable,
                                                            kernel_reg=self.use_kernel_reg,
                                                           freeze_attention=self.freeze_attention)

            if i == 0:
                downsampling["low_2"] = generate_residual_layer(layer_type= "NoAM", #self.s2f_att,
                                                                feature_filters=self.feature_filters,
                                                                nblocks=self.residual_nblocks,
                                                                name=f"Step{i}_ResidualLow2",
                                                                trainable=self.trainable,
                                                                kernel_reg=self.use_kernel_reg,
                                                                freeze_attention=self.freeze_attention)
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
            downsampling["low_3"] = generate_residual_layer(layer_type=self.f2s_att,
                                                            feature_filters=self.feature_filters,
                                                            nblocks=self.residual_nblocks,
                                                            name=f"Step{i}_ResidualLow3",
                                                            trainable=self.trainable,
                                                            kernel_reg=self.use_kernel_reg,
                                                           freeze_attention=self.freeze_attention)
            downsampling["up_2"] = layers.UpSampling2D(
                size=(2, 2),
                data_format=None,
                interpolation= "nearest", #"nearest",
                name=f"Step{i}_UpSampling2D",
                trainable=self.trainable,
            )
            downsampling["out"] = layers.Add(
                name=f"Step{i}_Add",
                trainable=self.trainable,
            )
        # endregion
        super().build(input_shape) 

    @classmethod
    def from_config(cls, config):    
        print("Restored Config:", config)  # Debugging output
        nfeatures = config["feature_filters"]
        feat_size = config["in_feature_size"]
        instance = cls(**config)  # Correctly restores the layer
        # Ensure layer is properly initialized by calling it once
        dummy_input = tf.zeros((1, feat_size, feat_size,nfeatures))  # Adjust shape as needed
        instance(dummy_input, training=False)  # This will trigger build()
        return instance