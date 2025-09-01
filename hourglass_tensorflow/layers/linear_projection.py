import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.activations import swish
from keras.saving import register_keras_serializable
from hourglass_tensorflow.layers.dummy_layers import zeroLayer
from hourglass_tensorflow.layers.sequential_layer import SequentialLayer


@register_keras_serializable(package="lLinearProj")
class LinearProjection(Layer):
    """
    This layer performs 2D convolution
    """
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        activation: str = None,
        kernel_initializer: str = "glorot_uniform",
        outmax: float = None,
        name: str = None,
        trainable: bool = True,
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.outmax = outmax
        # Create layers
        #self.conv = None
        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            name="Conv2D",
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            trainable=trainable,
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
                "outmax": self.outmax
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        # 我的Method
        #x = self.batch_norm(inputs,training=training)
        #x = self.conv(x)
        #return self.relu(x)
        #
        x = self.conv(inputs)     
        return x

    def build(self, input_shape):
        #print(f"[DEBUG]: {self.name} -- input shape : {input_shape}")
        super().build(input_shape) 
        self.built = True
    
    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """

@register_keras_serializable(package="lLinearProj")
class LinearProjectionLoRA(Layer):
    """
    This layer performs 2D convolution
    """
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        activation: str = None,
        kernel_initializer: str = "glorot_uniform",
        outmax: float = None,
        name: str = None,
        trainable: bool = True,
        activate_lora: bool = False,
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.outmax = outmax
        self.activate_lora = activate_lora 
        # Create layers
        #self.conv = None
        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            name="Conv2D",
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            trainable=trainable,
        )

        #lora path using a rank of 8
        self.lora = SequentialLayer(
            [       
                    layers.LayerNormalization(axis=-1
                                              ,epsilon=1e-6,
                                              name="lora_ln"),
                    layers.Conv2D(
                        filters=4,
                        kernel_size=1,
                        strides=self.strides,
                        padding=self.padding,
                        name="lora_a",
                        activation="gelu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                    ),

                    layers.Conv2D(
                        filters=self.filters,
                        kernel_size=1,
                        strides=self.strides,
                        padding=self.padding,
                        name="lora_b",
                        activation=None,
                        use_bias=True,
                        kernel_initializer="zeros",
                    )
                ],
                name="lora_path",
                trainable=self.trainable) if self.activate_lora else zeroLayer(output_channels=self.filters,
                                                                               name="lora_path",
                                                                               trainable = False)
        
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
                "outmax": self.outmax
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor: # training = True
        # 我的Method
        x = self.conv(inputs)    
        return x + self.lora(inputs)

    def build(self, input_shape):
        self.built = True
        if self.activate_lora and self.trainable:
            self.conv.trainable = False
            self.lora.trainable = True
        elif not self.activate_lora and self.trainable:
            self.lora.trainable = False
            self.conv.trainable = True
        else:
            self.lora.trainable = False
            self.conv.trainable = False
        super().build(input_shape) 
    
    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """