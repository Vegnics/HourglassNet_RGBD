import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.residual import ResidualLayer
#from hourglass_tensorflow.layers.residual_3 import ResidualLayerNoSkip as ResidualLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer
from hourglass_tensorflow.layers.conv_1 import Conv1Layer
from hourglass_tensorflow.layers.batch_norm_conv_1 import BatchNormConv1Layer


class HourglassLayerLast(Layer):
    def __init__(
        self,
        feature_filters: int = 256,# Number of feature maps
        output_filters: int = 16,  # Number of landmark heatmaps
        downsamplings: int = 4,    # Number of Downsamplings and upsamplings. 
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
        intermed = False
    ) -> None:
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        # Store Config
        self.downsamplings = downsamplings
        self.feature_filters = feature_filters
        self.output_filters = output_filters
        self.intermed = intermed
        # Init parameters
        self.layers = [{} for i in range(self.downsamplings)]
        # Create Layers
        #ConvBatchNormReluLayer
        self._hm_output = Conv1Layer(
            # Layer for heatmaps output.
            filters=output_filters,
            kernel_size=1,
            name="HeatmapOutput",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
        )
        """self._transit_output = BatchNormConv1Layer(
            # 
            filters=feature_filters,
            kernel_size=1,
            name="TransitOutput",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
         )
        """
        #self.transit_residual = ResidualLayer(output_filters=feature_filters,
        #                                    name="Last_residual",
        #                                    dtype=dtype,
        #                                    dynamic=dynamic,
        #                                    trainable=trainable,
        #                                    use_last_relu=True,
        #)
        self._last_residual = ResidualLayer(output_filters=feature_filters,
                                            name="Last_residual",
                                            dtype=dtype,
                                            dynamic=dynamic,
                                            trainable=trainable,
                                            use_last_relu=False,
        )
        self.relu = layers.ReLU(
            name="ReLU",
        )
        
        for i, downsampling in enumerate(self.layers):
            downsampling["up_1"] = ResidualLayer(
                output_filters=feature_filters,
                name=f"Step{i}_ResidualUp1",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["low_"] = layers.MaxPool2D(
                pool_size=(2, 2),
                padding="same",
                name=f"Step{i}_MaxPool",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["low_1"] = ResidualLayer(
                output_filters=feature_filters,
                name=f"Step{i}_ResidualLow1",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            if i == 0:
                downsampling["low_2"] = ResidualLayer(
                    output_filters=feature_filters,
                    name=f"Step{i}_ResidualLow2",
                    dtype=dtype,
                    dynamic=dynamic,
                    trainable=trainable,
                )
            downsampling["low_3"] = ResidualLayer(
                output_filters=feature_filters,
                name=f"Step{i}_ResidualLow3",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["up_2"] = layers.UpSampling2D(
                size=(2, 2),
                data_format=None,
                interpolation="nearest",
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

    def _recursive_call(self, input_tensor, step, training=True):
        step_layers = self.layers[step]
        up_1 = step_layers["up_1"](input_tensor, training=training)
        low_ = step_layers["low_"](input_tensor, training=training)
        low_1 = step_layers["low_1"](low_, training=training)
        if step == 0:
            low_2 = step_layers["low_2"](low_1, training=training)
        else:
            low_2 = self._recursive_call(low_1, step=(step - 1), training=training)
        low_3 = step_layers["low_3"](low_2, training=training)
        up_2 = step_layers["up_2"](low_3, training=training)
        out = step_layers["out"]([up_1, up_2], training=training)
        return out

    def call(self, inputs, training=True):
        x = self._recursive_call(
            input_tensor=inputs, step=self.downsamplings - 1, training=training
        )
        #intermediate = self._hm_output(x, training=training) # Intermediate Heatmap outputs >>>> IMPORTANT
        #_x = self.transit_residual(x,training=training)
        _out = self._last_residual(x,training=training)
        out_tensor = tf.add_n(
            [inputs, _out],name=f"{self.name}_OutputAdd",)
        out_tensor = self.relu(out_tensor)
        out = self._hm_output(out_tensor) 
        return out,out#, intermediate#tf.cast(tf.clip_by_value(tf.math.floor(intermediate),0.0,32767.0),dtype=tf.int16)
    def build(self, input_shape):
        pass