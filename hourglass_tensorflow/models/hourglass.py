import tensorflow as tf
from keras.models import Model

from hourglass_tensorflow.types.config import HTFModelAsLayers
from hourglass_tensorflow.layers.hourglass_Beta import HourglassLayer
#from hourglass_tensorflow.layers.hourglass import HourglassLayer
#from hourglass_tensorflow.layers.hourglass_mod import HourglassLayerLast
from hourglass_tensorflow.layers.downsampling import DownSamplingLayer
from hourglass_tensorflow.types.config.model import ATTENTION_MECHANISMS

class HourglassModel(Model):
    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 64,
        stages: int = 4,
        channel_number: int = 3,
        downsamplings_per_stage: int = 4,
        stage_filters: int = 256,
        channels_1joint: int = 16,
        channels_2joint: int = 16,
        intermediate_supervision: bool = True,
        name: str = None,
        trainable: bool = True,
        skip_AM: ATTENTION_MECHANISMS = "NoAM",
        s2f_AM: ATTENTION_MECHANISMS = "NoAM",
        f2s_AM: ATTENTION_MECHANISMS = "NoAM",
        use_2jointHM: bool = False,
        use_kernel_regularization: bool = False,
        freeze_attention_weights: bool = False,
        *args,
        **kwargs,
    )-> None:
        super().__init__(name=name,trainable=trainable)
        # Init
        self.channels_1J = channels_1joint
        self.channels_2J = channels_2joint
        self.intermediate_supervision = intermediate_supervision
        self.skip_AM = skip_AM
        self.s2f_AM = s2f_AM
        self.f2s_AM = f2s_AM
        self.stages = stages
        self.channelnum = channel_number
        self.use_2jointHM = use_2jointHM
        self.use_kernel_reg = use_kernel_regularization
        self.freeze_attention = freeze_attention_weights
        self.stage_filters = stage_filters
        self.input_size = input_size
        self.output_size = output_size
        self.ndownsamplings = downsamplings_per_stage
            #dtype=dtype,
            #dynamic=dynamic,
            #*args,
            #**kwargs,
        #)
        # Layers
        self.downsampling = None
        """
        DownSamplingLayer(
            input_size=input_size,
            output_size=output_size,
            kernel_size=7,
            output_filters=stage_filters,
            name="DownSampling",
            #dtype=dtype,
            #dynamic=dynamic,
            trainable=trainable,
        )
        """
        self.hourglasses = []
        """
            HourglassLayer(
                downsamplings=downsamplings_per_stage,
                feature_filters=stage_filters,
                joint_filters_1J=channels_1joint,
                joint_filters_2J=channels_2joint,
                name=f"Hourglass{i+1}",
                #dtype=dtype,
                #dynamic=dynamic,
                trainable=trainable,
                intermed= True,
                skip_attention = self.skip_AM,
                s2f_attention = self.s2f_AM,
                f2s_attention = self.f2s_AM,
                use_2jointHM = self.use_2jointHM,
                use_kernel_regularization=self.use_kernel_reg,
                freeze_attention = self.freeze_attention
            )
            for i in range(stages)
        ]
        """
        """
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
        """

    def _yes_no_str(self,val: bool):
        return "Yes" if val else "No"

    def build(self, input_shape = (None, 256, 256, 4)):
        # You can print the input shape to verify it
        print("---------Model configuration summary------------")
        print(f'Building model with input shape: {input_shape}')
        print(f"Attention mechanism in the Skip Layers: {self.skip_AM}")
        print(f"Attention mechanism in the S2F Layers (Bottom-up): {self.s2f_AM}")
        print(f"Attention mechanism in the F2S Layers (Top-down): {self.f2s_AM}")
        print(f"Use 2-Joint Heatmaps: {self._yes_no_str(self.use_2jointHM)}")
        print(f"Use kernel regularization: {self._yes_no_str(self.use_kernel_reg)}")
        print("------------------------------------------------")
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
    
    def get_config(self):
        return {
            **super().get_config(),
            **{
            "input_size":self.input_size,
            "output_size":self.output_size,
            "downsamplings_per_stage":self.ndownsamplings,
            "stages": self.stages,
            "channel_number":self.channelnum,
            "channels_1joint": self.channels_1J,
            "channels_2joint": self.channels_2J,
            "intermediate_supervision": self.intermediate_supervision,
            "skip_AM": self.skip_AM,
            "s2f_AM": self.s2f_AM,
            "f2s_AM": self.f2s_AM,
            "use_2jointHM": self.use_2jointHM,
            "use_kernel_regularization": self.use_kernel_reg,
            "freeze_attention_weights":self.freeze_attention,
            "stage_filters":self.stage_filters
            },
        }

    @classmethod
    def from_config(cls, config):
        print("Restored Config [HOURGLASS MODEL]:", config)  # Debugging output
        return cls(**config)

    def build(self, input_shape):
        # Layers
        self.downsampling = DownSamplingLayer(
            input_size=self.input_size,
            output_size=self.output_size,
            kernel_size=7,
            output_filters=self.stage_filters,
            name="DownSampling",
            #dtype=dtype,
            #dynamic=dynamic,
            trainable=self.trainable,
        )
        self.hourglasses = [
            HourglassLayer(
                downsamplings=self.ndownsamplings,
                feature_filters=self.stage_filters,
                joint_filters_1J=self.channels_1J,
                joint_filters_2J=self.channels_2J,
                name=f"Hourglass{i+1}",
                #dtype=dtype,
                #dynamic=dynamic,
                trainable=self.trainable,
                intermed= True,
                skip_attention = self.skip_AM,
                s2f_attention = self.s2f_AM,
                f2s_attention = self.f2s_AM,
                use_2jointHM = self.use_2jointHM,
                use_kernel_regularization=self.use_kernel_reg,
                freeze_attention = self.freeze_attention
            )
            for i in range(self.stages)
        ]
        super().build(input_shape) 
        self.built = True



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
        #dtype=dtype,
        #dynamic=dynamic,
        trainable=trainable,
    )
    hourglasses = [
        HourglassLayer(
            downsamplings=downsamplings_per_stage,
            feature_filters=stage_filters,
            output_filters=output_channels,
            name=f"Hourglass{i+1}",
            #dtype=dtype,
            #dynamic=dynamic,
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
