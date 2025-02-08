import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.residual import ResidualLayer
from hourglass_tensorflow.layers.residual_input import ResidualLayerIn
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer
from hourglass_tensorflow.layers.hm_output import HMOut
from hourglass_tensorflow.layers.batch_norm_conv_1 import BatchNormConv1Layer
from hourglass_tensorflow.layers.residual_with_attention import ResidualLayerAttention
from hourglass_tensorflow.layers.residual_with_attention_spatial import ResidualLayerAttentionSpatial

def generate_residual_layer(layer_type: str ,
                            feature_filters: int ,
                            name: str = None ,
                            #dtype = None ,
                            #dynamic = False,
                            trainable = True,
                            kernel_reg = False,
                            freeze_attention=False):
        
        if layer_type == "NoAM":
            return  ResidualLayer(
                output_filters= feature_filters,
                name=name,
                #dtype=dtype,
                #dynamic=dynamic,
                trainable=trainable,)
        elif layer_type == "SAM":
            return ResidualLayerAttentionSpatial(
                output_filters=feature_filters,
                name=name,
                #dtype=dtype,
                #dynamic=dynamic,
                trainable=trainable,
                kernel_reg=kernel_reg,
                freeze_attention=freeze_attention,)
        elif layer_type == "FAM":
            return ResidualLayerAttention(
                output_filters=feature_filters,
                name=name,
                #dtype=dtype,
                #dynamic=dynamic,
                trainable=trainable,
                kernel_reg = kernel_reg,
                freeze_attention=freeze_attention)
        else:
            raise Exception(f"The residual layer type: {layer_type} is invalid.")

class ResidualInWithBNRC(Layer):
    def __init__(
        self,
        output_filters: int = 256,# Number of feature maps
        kernel_initializer: str = "glorot_uniform",
        momentum: float = 0.9,
        epsilon: float = 0.0001,
        name: str = None,
        #dtype=None,
        #dynamic=False,
        trainable: bool = True,
    ) -> None:
        #super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        super().__init__(name=name,trainable=trainable)
        # Store Config
        self.feature_filters = output_filters

        self.conv = layers.Conv2D(
            filters=output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name="Conv2D",
            activation=None,
            kernel_initializer=kernel_initializer,
        )

        self.batch_norm = layers.BatchNormalization(
            axis=-1,
            momentum=momentum,
            epsilon=epsilon,
            trainable=trainable,
            name="BatchNorm",
        ) 

        self.relu = layers.ReLU(
            name="ReLU",
        )

        self.residual1 =  ResidualLayerIn(
                                            output_filters=output_filters,
                                            name="ResidualInCBNR",
                                            #dtype=dtype,
                                            #dynamic=dynamic,
                                            trainable=trainable,
                                            epsilon=0.001,
                                            momentum=0.97,
        )
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "feature_filters": self.feature_filters,
            },
        }
    def call(self,inputs, training=True):
        x = self.residual1(inputs, training=training)
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        x = self.conv(x)
        return x


class zeroLayer(Layer):
    def __init__(
        self,
        output_channels: int = 16,# Number of feature maps
        name: str = None,
    ) -> None:
        super().__init__(name=name, trainable=False)
        # Store Config
        self.output_channels = output_channels 
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_channels": self.output_channels,
            },
        }
    def call(self,inputs):
        x = tf.reduce_sum(0.0*inputs,axis=3)
        x = tf.expand_dims(x,axis=3)
        return x*tf.zeros(shape=(1,1,1,self.output_channels))

class ResidualWithBNRC(Layer):
    def __init__(
        self,
        output_filters: int = 256,# Number of feature maps
        kernel_initializer: str = "glorot_uniform",
        momentum: float = 0.9,
        epsilon: float = 0.0001,
        name: str = None,
        #dtype=None,
        #dynamic=False,
        trainable: bool = True,
        attention: str = None,
    ) -> None:
        #super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        super().__init__(name=name, trainable=trainable)
        # Store Config
        self.feature_filters = output_filters

        self.conv = layers.Conv2D(
            filters=output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name="Conv2D",
            activation=None,
            kernel_initializer=kernel_initializer,
        )

        self.batch_norm = layers.BatchNormalization(
            axis=-1,
            momentum=momentum,
            epsilon=epsilon,
            trainable=trainable,
            name="BatchNorm",
        ) 

        self.relu = layers.ReLU(
            name="ReLU",
        )
        self.residual1 =  generate_residual_layer(layer_type=attention,
                                                feature_filters=output_filters,
                                                name="Residual",
                                                #dtype=dtype,
                                                #dynamic=dynamic,
                                                trainable=trainable,)
    
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "feature_filters": self.feature_filters,
            },
        }
    def call(self,inputs, training=True):
        x = self.residual1(inputs, training=training)
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        x = self.conv(x)
        return x
 
class HourglassLayer(Layer):
    def __init__(
        self,
        feature_filters: int = 256,# Number of feature maps
        joint_filters_1J: int = 16,  # Number of 1-joint heatmaps
        joint_filters_2J: int = 16,  # Number of 2-joint heatmaps
        downsamplings: int = 4,    # Number of Downsamplings and upsamplings. 
        name: str = None,
        #dtype=None,
        #dynamic: bool =False,
        trainable: bool = True,
        intermed: bool = False,
        use_2jointHM: bool = False,
        skip_attention: str = None,
        s2f_attention: str = None,
        f2s_attention: str = None,
        use_kernel_regularization: bool = False,
        freeze_attention: bool = False
    ) -> None:
        #super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        super().__init__(name=name, trainable=trainable)
        # Store Config
        self.downsamplings = downsamplings
        self.feature_filters = feature_filters
        self.joint_filters_1J= joint_filters_1J
        self.joint_filters_2J= joint_filters_2J
        self.intermed = intermed
        self.trainable = trainable
        self.skip_att = skip_attention
        self.s2f_att = s2f_attention
        self.f2s_att = f2s_attention
        # Init parameters
        self.layers = [{} for i in range(self.downsamplings)]
        # Create Layers
        #ConvBatchNormReluLayer
        self._hm1_output = HMOut(
            # Layer for heatmaps output.
            filters=joint_filters_1J, #output_filters
            kernel_size=1,
            name="HeatmapOutput",
            #dtype=dtype,
            #dynamic=dynamic,
            trainable=trainable,
            outmax=None,
        )

        #"""
        self._hm2_output = HMOut(
            # Layer for heatmaps output.
            filters=joint_filters_2J,#14,
            kernel_size=1,
            name="Heatmap2Output",
            #dtype=dtype,
            #dynamic=dynamic,
            trainable=trainable,
            outmax=None,
        ) if use_2jointHM else zeroLayer(joint_filters_2J,name="ZeroHeatmap2")
        #"""


        self._residual_2j = ResidualLayerIn(output_filters=feature_filters,
                                            name="Transit_Output",
                                            #dtype=dtype,
                                            #dynamic=dynamic,
                                            trainable=trainable,
                                            epsilon=0.001,
                                            momentum=0.97,
        ) if use_2jointHM else zeroLayer(feature_filters,name="ZeroResidual2J")
        
        self._merge_feats_main = BatchNormConv1Layer(filters=feature_filters,
                                            kernel_size=1,
                                            name="Merge_Feats",
                                            #dtype=dtype,
                                            #dynamic=dynamic,
                                            trainable=trainable,
        )

        self._merge_feats_1j = BatchNormConv1Layer(filters=feature_filters,
                                            kernel_size=1,
                                            name="Merge_Feats",
                                            #dtype=dtype,
                                            #dynamic=dynamic,
                                            trainable=trainable,
        )

        #self.relu = layers.ReLU(
        #    name="ReLU",
        #)
        
        for i, downsampling in enumerate(self.layers):
            downsampling["up_1"] = generate_residual_layer(layer_type=self.skip_att,feature_filters=self.feature_filters,
                                                           name=f"Step{i}_ResidualUp1",dtype=self.dtype,dynamic=self.dynamic,trainable=self.trainable,kernel_reg=use_kernel_regularization,freeze_attention=freeze_attention)
            downsampling["low_"] = layers.MaxPool2D(
                pool_size=(2, 2),
                padding="valid",
                name=f"Step{i}_MaxPool",
                #dtype=dtype,
                #dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["low_1"] = generate_residual_layer(layer_type=self.s2f_att,
                                                            feature_filters=self.feature_filters,
                                                            name=f"Step{i}_ResidualLow1",
                                                            #dtype=self.dtype,
                                                            #dynamic=self.dynamic,
                                                            trainable=self.trainable,
                                                            kernel_reg=use_kernel_regularization,
                                                            freeze_attention=freeze_attention)

            if i == 0:
                downsampling["low_2"] = generate_residual_layer(layer_type=self.s2f_att,
                                                                feature_filters=self.feature_filters,
                                                                name=f"Step{i}_ResidualLow2",
                                                                #dtype=self.dtype,
                                                                #dynamic=self.dynamic,
                                                                trainable=self.trainable,
                                                                kernel_reg=use_kernel_regularization,
                                                                freeze_attention=freeze_attention)
            elif i == 3:
                downsampling["low_in"] =  generate_residual_layer(layer_type=self.s2f_att,
                                                                  feature_filters=self.feature_filters,
                                                                  name=f"Step{i}_ResidualMainIn",
                                                                  #dtype=self.dtype,
                                                                  #dynamic=self.dynamic,
                                                                  trainable=self.trainable,
                                                                  kernel_reg=use_kernel_regularization,
                                                                  freeze_attention=freeze_attention)
                downsampling["low_out"] = ResidualWithBNRC(
                    output_filters=feature_filters,
                    name=f"Step{i}_ResidualMainOut",
                    #dtype=dtype,
                    #dynamic=dynamic,
                    trainable=trainable,
                    attention=self.f2s_att,
                )
            downsampling["low_3"] = generate_residual_layer(layer_type=self.f2s_att,
                                                            feature_filters=self.feature_filters,
                                                            name=f"Step{i}_ResidualLow3",
                                                            #dtype=self.dtype,
                                                            #dynamic=self.dynamic,
                                                            trainable=self.trainable,
                                                            kernel_reg=use_kernel_regularization,
                                                            freeze_attention=freeze_attention)
            downsampling["up_2"] = layers.UpSampling2D(
                size=(2, 2),
                data_format=None,
                interpolation= "nearest", #"nearest",
                name=f"Step{i}_UpSampling2D",
                #dtype=dtype,
                #dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["out"] = layers.Add(
                name=f"Step{i}_Add",
                #dtype=dtype,
                #dynamic=dynamic,
                trainable=trainable,
            )
        # endregion

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "downsamplings": self.downsamplings,
                "feature_filters": self.feature_filters,
                "joints_filters_1": self.joint_filters_1J,
                "joints_filters_2": self.joint_filters_2J,
            },
        }
        
    def _recursive_call(self, input_tensor, step, training=True):
        step_layers = self.layers[step]
        _input = input_tensor
        if step == 3:
            _input = step_layers["low_in"](_input, training=training)
        up_1 = step_layers["up_1"](_input, training=training)
        low_ = step_layers["low_"](_input, training=training)
        low_1 = step_layers["low_1"](low_, training=training)
        if step == 0:
            low_2 = step_layers["low_2"](low_1, training=training)
        else:
            low_2 = self._recursive_call(low_1, step=(step - 1), training=training)
        low_3 = step_layers["low_3"](low_2, training=training)
        up_2 = step_layers["up_2"](low_3, training=training)
        out = step_layers["out"]([up_1, up_2], training=training)
        if step == 3:
            out = step_layers["low_out"](out,training=training)
        return out

    def call(self, inputs, training=False):
        _x = self._recursive_call(
            input_tensor=inputs, step=self.downsamplings - 1, training=training
        )
        main_feats = self._merge_feats_main(_x)
        intermediate_2jhms = self._hm2_output(_x,training=training) 
        transit_2jhms = self._residual_2j(intermediate_2jhms, training=training)

        intermediate_1jhms = self._hm1_output(tf.add_n([_x,transit_2jhms]), training=training)
        #bpart_feats = self.body_part_residual(intermediate_2jhms,training=training)
        #intermediate_1jhms = self._hm_output(tf.add_n([_x,bpart_feats]), training=training) # Intermediate Heatmap outputs >>>> IMPORTANT
        #intermediate_1jhms = self._hm_output(tf.add_n([_x,bpart_feats]), training=training)

        #_out = self._last_residual(_x,training=training)
        out_tensor = tf.add_n(
            [inputs, main_feats, self._merge_feats_1j(intermediate_1jhms)], #_out
            name=f"{self.name}_OutputAdd",
        )
        #return self.relu(out_tensor), intermediate#tf.cast(tf.clip_by_value(tf.math.floor(intermediate),0.0,32767.0),dtype=tf.int16)
        
        return out_tensor, tf.concat([intermediate_1jhms,intermediate_2jhms],axis=-1)
        #return out_tensor, intermediate_1jhms 
    def build(self, input_shape):
        pass
