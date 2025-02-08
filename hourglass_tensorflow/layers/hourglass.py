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


class ResidualInWithBNRC(Layer):
    def __init__(
        self,
        output_filters: int = 256,# Number of feature maps
        kernel_initializer: str = "glorot_uniform",
        momentum: float = 0.9,
        epsilon: float = 0.0001,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
    ) -> None:
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
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

        self.residual1 =  ResidualLayerIn( #AttentionSpatial(
                                            output_filters=output_filters,
                                            name="ResidualInCBNR",
                                            dtype=dtype,
                                            dynamic=dynamic,
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

class ResidualWithBNRC(Layer):
    def __init__(
        self,
        output_filters: int = 256,# Number of feature maps
        kernel_initializer: str = "glorot_uniform",
        momentum: float = 0.9,
        epsilon: float = 0.0001,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
    ) -> None:
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
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

        self.residual1 =  ResidualLayerAttentionSpatial( #AttentionSpatial(
                                            output_filters=output_filters,
                                            name="Residual",
                                            dtype=dtype,
                                            dynamic=dynamic,
                                            trainable=trainable,
                                            use_last_relu=False,
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
 
class HourglassLayer(Layer):
    def __init__(
        self,
        feature_filters: int = 256,# Number of feature maps
        output_filters: int = 16,  # Number of landmark heatmaps
        downsamplings: int = 4,    # Number of Downsamplings and upsamplings. 
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
        intermed = False,
        skip_attention: str = None,
        s2f_attention: str = None,
        f2s_attention: str = None
    ) -> None:
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        # Store Config
        self.downsamplings = downsamplings
        self.feature_filters = feature_filters
        self.output_filters = output_filters
        self.intermed = intermed
        self.trainable = trainable
        self.dtype  = dtype
        # Init parameters
        self.layers = [{} for i in range(self.downsamplings)]
        # Create Layers
        #ConvBatchNormReluLayer
        self._hm1_output = HMOut(
            # Layer for heatmaps output.
            filters=14, #output_filters
            kernel_size=1,
            name="HeatmapOutput",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
            outmax=None,
        )

        #"""
        self._hm2_output = HMOut(
            # Layer for heatmaps output.
            filters= 12,#14,
            kernel_size=1,
            name="Heatmap2Output",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
            outmax=None,
        )
        #"""


        self._residual_2j = ResidualLayerIn(output_filters=feature_filters,
                                            name="Transit_Output",
                                            dtype=dtype,
                                            dynamic=dynamic,
                                            trainable=trainable,
                                            epsilon=0.001,
                                            momentum=0.97,
        )
        
        self._merge_feats_main = BatchNormConv1Layer(filters=feature_filters,
                                            kernel_size=1,
                                            name="Merge_Feats",
                                            dtype=dtype,
                                            dynamic=dynamic,
                                            trainable=trainable,
        )

        self._merge_feats_1j = BatchNormConv1Layer(filters=feature_filters,
                                            kernel_size=1,
                                            name="Merge_Feats",
                                            dtype=dtype,
                                            dynamic=dynamic,
                                            trainable=trainable,
        )

        #self.relu = layers.ReLU(
        #    name="ReLU",
        #)
        
        for i, downsampling in enumerate(self.layers):
            downsampling["up_1"] = ResidualLayer( #Spatial
                output_filters=feature_filters,
                name=f"Step{i}_ResidualUp1",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["low_"] = layers.MaxPool2D(
                pool_size=(2, 2),
                padding="valid",
                name=f"Step{i}_MaxPool",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["low_1"] = ResidualLayerAttention( #Attention( #Spatial #Attention
                output_filters=feature_filters,
                name=f"Step{i}_ResidualLow1",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            if i == 0:
                downsampling["low_2"] = ResidualLayerAttention( #Attention(
                    output_filters=feature_filters,
                    name=f"Step{i}_ResidualLow2",
                    dtype=dtype,
                    dynamic=dynamic,
                    trainable=trainable,
                )
            elif i == 3:
                downsampling["low_in"] =  ResidualLayerAttention( #Attention( #Spatial #Attention
                    output_filters=feature_filters,
                    name=f"Step{i}_ResidualMainIn",
                    dtype=dtype,
                    dynamic=dynamic,
                    trainable=trainable
                )
                downsampling["low_out"] = ResidualWithBNRC( #ResidualWithCBNR( #Attention #Spatial
                    output_filters=feature_filters,
                    name=f"Step{i}_ResidualMainOut",
                    dtype=dtype,
                    dynamic=dynamic,
                    trainable=trainable,
                )
            downsampling["low_3"] = ResidualLayerAttentionSpatial( #AttentionSpatial( #Attention #Spatial
                output_filters=feature_filters,
                name=f"Step{i}_ResidualLow3",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["up_2"] = layers.UpSampling2D(
                size=(2, 2),
                data_format=None,
                interpolation= "nearest", #"nearest",
                name=f"Step{i}_UpSampling2D",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["out"] = layers.Add(
                name=f"Step{i}_Add",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            #downsampling["out_"] = ResidualLayerAttention(
            #    output_filters=feature_filters,
            #    name=f"Step{i}_ResidualAttention",
            #    dtype=dtype,
            #    dynamic=dynamic,
            #    trainable=trainable,
            #)
        # endregion

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "downsamplings": self.downsamplings,
                "feature_filters": self.feature_filters,
                "output_filters": self.output_filters,
            },
        }
    def generate_residual_layer(self,layer_type="residual"):
        if layer_type == "residual":
            return ResidualLayer()

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
        self.built = True