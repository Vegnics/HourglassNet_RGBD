from os import name
import tensorflow as tf
from keras.models import Model
from keras.utils import register_keras_serializable
from keras.layers import Lambda,Layer
from typing import List
from keras import Input as InputTensor
from hourglass_tensorflow.types.config import HTFModelAsLayers
from hourglass_tensorflow.layers.hourglass_lora import HourglassLayerLora
from hourglass_tensorflow.layers.downsampling_lora import DownSamplingLayerLora
from hourglass_tensorflow.types.config.model import ATTENTION_MECHANISMS

@register_keras_serializable(package="cHourglassModelLora")
class WrappedModel(Model):
    def __init__(self, model, **kwargs):
        super(WrappedModel, self).__init__(**kwargs)
        # Instantiate your core functional model here.
        # This will be the "core" of our model.
        self.core_model = model

    def call(self, inputs):
        # The call method simply passes the inputs to the core functional model.
        return self.core_model(inputs)

    # This method is correctly overridden on the class.
    # It will be executed by model.fit().
    def train_step(self, data):
        # Unpack the data received from model.fit().
        x, y_true = data
        
        # A tf.print() here will now work as expected.
        tf.print("y_true in train_step (Eager Execution):", y_true)
        
        with tf.GradientTape() as tape:
            # Pass y_true explicitly to the call method if needed.
            y_pred = self(x, training=True)
            
            # Get the main loss. Use self.losses for regularization losses.
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # Compute gradients and apply them.
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update and return metrics.
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        # Override get_config to serialize the functional model's configuration.
        config = super(WrappedModel, self).get_config()
        config['core_config'] = self.core_model.get_config()
        return config

    @classmethod
    def from_config(cls, config):
        core_config = config.pop('core_config')
        # Create an instance and then restore the core model from its config.
        instance = cls(**config)
        instance.core_model = Model.from_config(core_config)
        return instance


@register_keras_serializable(package="cHourglassModelLora")
class stack_tensors_layer(Layer):
    def __init__(
        self,
        name: str = None,
        trainable: bool = False,
        **kwargs,
    )-> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
    def call(self, inputs: List[tf.Tensor], training=True):
        return tf.stack(inputs, axis=1)
    def build(self, input_shape):
        super().build(input_shape)
    

@register_keras_serializable(package="cHourglassModelLora")
class HourglassModelLora(Model):
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
        residual_nblocks: int = None,
        activate_lora: bool = False,
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
        self.residual_nblocks = residual_nblocks
        self.trainable = trainable
        self.activate_lora = activate_lora

        """
        Layers block
        """
        # Downsampling layer
        self.downsampling = DownSamplingLayerLora(
            input_size=self.input_size,
            output_size=self.output_size,
            kernel_size=7,
            output_filters=self.stage_filters,
            name="DownSampling",
            residual_nblocks=self.residual_nblocks,
            trainable=self.trainable,
            activate_lora = self.activate_lora
        )
        
        # Hourglass layers
        self.hourglasses = [
            HourglassLayerLora(
                downsamplings=self.ndownsamplings,
                feature_filters=self.stage_filters,
                joint_filters_1J=self.channels_1J,
                joint_filters_2J=self.channels_2J,
                name=f"Hourglass{i+1}",
                trainable=self.trainable,
                intermed= True,
                skip_attention = self.skip_AM,
                s2f_attention = self.s2f_AM,
                f2s_attention = self.f2s_AM,
                use_2jointHM = self.use_2jointHM,
                use_kernel_regularization=self.use_kernel_reg,
                freeze_attention = self.freeze_attention,
                residual_nblocks=self.residual_nblocks,
                activate_lora = self.activate_lora
            )
            for i in range(self.stages)
        ]

        # Set the hourglasses as attributes for serialization
        for k,hg in enumerate(self.hourglasses):
            self.__setattr__(f"HG_{k}",hg)
        self.stacker = stack_tensors_layer(name="StackedOutput",trainable=False)
    
    def _yes_no_str(self,val: bool):
        return "Yes" if val else "No"
    
    def wrap_model(self):
        inputs = InputTensor(shape=(self.input_size, self.input_size, self.channelnum))
        outputs = self.call(inputs)
        # Intermediate output for axuliary loss
        #aux_out = self.get_layer('Hourglass2').output[2]
        wrapped = Model(inputs, outputs,name="FunctionalHourglassModel")
        wrapped.core = self
        original_get_config = wrapped.get_config

        def merged_get_config():
            config = original_get_config()
            config["core_config"] = self.get_config()
            return config

        wrapped.get_config = merged_get_config
        return wrapped


    def build(self, input_shape = (None, 256, 256, 4)):
        super().build(input_shape)
        self.built = True
        # You can print the input shape to verify it
        print("---------Model configuration summary------------")
        print(f'Building model with input shape: {input_shape}')
        print(f"Number of features: {self.stage_filters}")
        print(f"Residual blocks: {self.residual_nblocks}")
        print(f"Attention mechanism in the Skip Layers: {self.skip_AM}")
        print(f"Attention mechanism in the S2F Layers (Bottom-up): {self.s2f_AM}")
        print(f"Attention mechanism in the F2S Layers (Top-down): {self.f2s_AM}")
        print(f"Use 2-Joint Heatmaps: {self._yes_no_str(self.use_2jointHM)}")
        print(f"Use kernel regularization: {self._yes_no_str(self.use_kernel_reg)}")
        print("------------------------------------------------")

    #def call(self, inputs: tf.Tensor, training=True):
    def call(self, inputs, y_true=None, training=None):
        tf.print(y_true)
        x = self.downsampling(inputs,training=training)
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
            x, y,_ = hglayer(x,y_true,training=training) # x is the output features, y is the intermediate output heatmaps
            outputs_list.append(y)
        outputs = self.stacker(outputs_list)
        return outputs
    
    
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
            "stage_filters":self.stage_filters,
            "residual_nblocks":self.residual_nblocks,
            "activate_lora":self.activate_lora
            },
        }
    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        print("Restored Config [Hourglass Model]:", config)  # Debugging output
        return instance
    """
    
@register_keras_serializable(package="cHourglassModelLora")
def build_hourglassModelLora(
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
        residual_nblocks: int = None,
        activate_lora: bool = False,
        ):
    
    input_shape=(input_size,input_size, channel_number)
    downsampling = DownSamplingLayerLora(
            input_size=input_size,
            output_size=output_size,
            kernel_size=7,
            output_filters=stage_filters,
            name="DownSampling",
            residual_nblocks=residual_nblocks,
            trainable=trainable,
            activate_lora = activate_lora,
        )
    hourglasses = [
            HourglassLayerLora(
                downsamplings=downsamplings_per_stage,
                feature_filters=stage_filters,
                joint_filters_1J=channels_1joint,
                joint_filters_2J=channels_2joint,
                name=f"Hourglass{i+1}",
                trainable=trainable,
                intermed= True,
                skip_attention = skip_AM,
                s2f_attention = s2f_AM,
                f2s_attention = f2s_AM,
                use_2jointHM = use_2jointHM,
                use_kernel_regularization=use_kernel_regularization,
                freeze_attention = freeze_attention_weights,
                residual_nblocks= residual_nblocks,
                activate_lora = activate_lora,
            )
            for i in range(stages)
        ]
    stacker = stack_tensors_layer(name="StackedOutput",trainable=False)
    inputs = InputTensor(shape=input_shape, name="InputHG")
    outputs_list = []
    x = inputs
    x  = downsampling(x)
    for _layer in hourglasses:
        x,y,_ = _layer(x)
        outputs_list.append(y)

    outputs = stacker(outputs_list)
    return Model(inputs, outputs, name=name)

@register_keras_serializable(package="cHourglassModelLora")
def model_as_layers_lora(
    inputs: tf.Tensor,
    input_size: int = 256,
    output_size: int = 64,
    stages: int = 4,
    downsamplings_per_stage: int = 4,
    stage_filters: int = 256,
    output_channels: int = 16,
    intermediate_supervision: bool = True,
    name: str = None,
    trainable: bool = True,
    skip_AM: ATTENTION_MECHANISMS = "NoAM",
    s2f_AM: ATTENTION_MECHANISMS = "NoAM",
    f2s_AM: ATTENTION_MECHANISMS = "NoAM",
    use_2jointHM: bool = False,
    use_kernel_regularization: bool = False,
    freeze_attention_weights: bool = False,
    residual_nblocks: int = None,
    activate_lora: bool = False,
    *args,
    **kwargs,
) -> HTFModelAsLayers:
    downsampling = DownSamplingLayerLora(
        input_size=input_size,
        output_size=output_size,
        kernel_size=7,
        output_filters=stage_filters,
        name="DownSampling",
        residual_nblocks= residual_nblocks,
        trainable= trainable,
        activate_lora = activate_lora
    )
    hourglasses = [
        HourglassLayerLora(
            downsamplings=downsamplings_per_stage,
            feature_filters=stage_filters,
            output_filters=output_channels,
            name=f"Hourglass{i+1}",
            trainable=trainable,
            intermed= True,
            skip_attention = skip_AM,
            s2f_attention = s2f_AM,
            f2s_attention = f2s_AM,
            use_2jointHM = use_2jointHM,
            residual_nblocks=residual_nblocks,
            activate_lora = activate_lora
        )
        for i in range(stages)
    ]

    x = downsampling(inputs)
    output_list = []
    for layer in hourglasses:
        x, y,_ = layer(x)
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
