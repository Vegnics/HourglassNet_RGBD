import tensorflow as tf
from keras.models import Model

from hourglass_tensorflow.types.config import HTFModelAsLayers
from hourglass_tensorflow.layers.hourglass import HourglassLayer
from hourglass_tensorflow.layers.hourglass_mod import HourglassLayerLast
from hourglass_tensorflow.layers.downsampling import DownSamplingLayer


class HourglassModel(Model):
    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 64,
        stages: int = 4,
        channel_number: int = 3,
        downsamplings_per_stage: int = 4,
        stage_filters: int = 256,
        output_channels: int = 16,
        intermediate_supervision: bool = True,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
            *args,
            **kwargs,
        )
        # Init
        self._intermediate_supervision = intermediate_supervision

        # Layers
        self.downsampling = DownSamplingLayer(
            input_size=input_size,
            output_size=output_size,
            kernel_size=7,
            output_filters=stage_filters,
            name="DownSampling",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
        )
        self.hourglasses = [
            HourglassLayer(
                downsamplings=downsamplings_per_stage,
                feature_filters=stage_filters,
                output_filters=output_channels,
                name=f"Hourglass{i+1}",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
                intermed= True 
            )
            for i in range(stages-1)
        ]
        self.hourglasses.append(HourglassLayerLast(
                downsamplings=downsamplings_per_stage,
                feature_filters=stage_filters,
                output_filters=output_channels,
                name=f"Hourglass_LAST",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
                intermed= False 
            )
        )

    def build(self, input_shape = (None, 256, 256, 4)):
        # You can print the input shape to verify it
        print(f'Building model with input shape: {input_shape}')
        # Optionally, you can use the input_shape to dynamically define layers
        super(HourglassModel, self).build(input_shape)

    def call(self, inputs: tf.Tensor, training=True):
        x = self.downsampling(tf.cast(inputs,dtype=tf.dtypes.float32))
        #x = self.downsampling(inputs)
        outputs_list = []
        """
        for layer in self.hourglasses:
            x, y = layer(x) # x is the output features, y is the intermediate output heatmaps
            if self._intermediate_supervision:
                outputs_list.append(y)
        if self._intermediate_supervision:
            self._outputs = tf.stack(outputs_list, axis=1, name="NetworkStackedOutput")
        else:
            self._outputs = y
        return self._outputs
        """
        for hglayer in self.hourglasses:
            x, y = hglayer(x) # x is the output features, y is the intermediate output heatmaps
            outputs_list.append(y)
        self._outputs = tf.stack(outputs_list, axis=1, name="NetworkStackedOutput")
        return self._outputs


def model_as_layers(
    inputs: tf.Tensor,
    input_size: int = 256,
    output_size: int = 64,
    stages: int = 4,
    downsamplings_per_stage: int = 4,
    stage_filters: int = 256,
    output_channels: int = 16,
    intermediate_supervision: bool = True,
    name: str = None,
    dtype=None,
    dynamic=False,
    trainable: bool = True,
    *args,
    **kwargs,
) -> HTFModelAsLayers:
    downsampling = DownSamplingLayer(
        input_size=input_size,
        output_size=output_size,
        kernel_size=7,
        output_filters=stage_filters,
        name="DownSampling",
        dtype=dtype,
        dynamic=dynamic,
        trainable=trainable,
    )
    hourglasses = [
        HourglassLayer(
            downsamplings=downsamplings_per_stage,
            feature_filters=stage_filters,
            output_filters=output_channels,
            name=f"Hourglass{i+1}",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
        )
        for i in range(stages)
    ]

    x = downsampling(inputs)
    output_list = []
    for layer in hourglasses:
        x, y = layer(x)
        if intermediate_supervision:
            output_list.append(y)
    if intermediate_supervision:
        outputs = tf.stack(output_list, axis=1, name="NetworkStackedOutput")
    else:
        outputs = y

    model = Model(inputs=inputs, outputs=outputs)

    return HTFModelAsLayers(
        downsampling=downsampling, hourglasses=hourglasses, outputs=outputs, model=model
    )
