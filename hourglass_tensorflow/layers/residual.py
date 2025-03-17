import tensorflow as tf
from keras import layers
from keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

from hourglass_tensorflow.layers.skip import SkipLayer
from hourglass_tensorflow.layers.conv_block import ConvBlockLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer
from hourglass_tensorflow.layers.dummy_layers import IdentityLayer

@register_keras_serializable(package="lResiduals")
class ResidualBlock(Layer):
    def __init__(
        self,
        output_filters: int,
        momentum: float = 0.97,
        epsilon: float = 0.001,
        name: str = None,
        trainable: bool = True,
        use_last_relu: bool = False,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        # Convolutional block
        self.conv_block = None
        self.add = None
        self.relu= None

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_last_relu": self.use_last_relu,
                "trainable": self.trainable
            },
        }
    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        _sum = self.add(
            [
                self.conv_block(inputs, training=training),
                inputs,
            ])
        return self.relu(_sum)
    
    def build(self, input_shape):
        print(f"[DEBUG]: {self.name} -- input shape : {input_shape}")
        self.conv_block = ConvBlockLayer(
            output_filters=self.output_filters,
            momentum=self.momentum,
            epsilon=self.epsilon,
            name="ConvBlock",
            trainable=self.trainable,
        )
        self.add = layers.Add(name="Add")
        self.relu= layers.ReLU(name="ReLu_identity",)  if self.use_last_relu else IdentityLayer(name="ReLu_identity")
        super().build(input_shape)
        self.built = True

    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        # Set attributes that are dynamically created and should not be part of the config.
        instance.conv_block = None  # Will be initialized in build() method
        instance.add = None
        instance.relu = None
        return instance

@register_keras_serializable(package="lResiduals")
class ResidualLayer(Layer):
    def __init__(
        self,
        output_filters: int,
        nblocks: int,
        momentum: float = 0.97,
        epsilon: float = 0.001,
        name: str = None,
        trainable: bool = True,
        use_last_relu: bool = False,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        self.nblocks = nblocks
        self.residual_blocks = []
        self.trainable = trainable

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_last_relu": self.use_last_relu,
                "nblocks": self.nblocks,
                "trainable": self.trainable,
            },
        }
    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = 1.0*inputs
        for layer in self.residual_blocks:
            x = layer(x,training=training)
        return x
    
    def build(self, input_shape):
        print(f"[DEBUG]: {self.name} -- input shape : {input_shape}")
        self.residual_blocks = [ResidualBlock(output_filters= self.output_filters,
                                            name=f"{self.name}_block{k}",
                                            use_last_relu = self.use_last_relu,
                                            trainable=self.trainable) for k in range(self.nblocks)]
        super().build(input_shape)
        self.built = True
    
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        instance.residual_blocks = []
        return instance
    
@register_keras_serializable(package="lResiduals")
class ResidualBlockIn(Layer):
    def __init__(
        self,
        output_filters: int,
        momentum: float = 0.98,
        epsilon: float = 0.001,
        name: str = None,
        trainable: bool = True,
        use_last_relu: bool = False,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        # Input layer (used just to match the dimensionality)
        self.match_layer =  None
        # Convolutional block
        self.conv_block = None

        self.add = None
        self.relu= None
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_last_relu": self.use_last_relu,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        _inputs = self.match_layer(inputs,training=training)
        _sum = self.add(
            [
                self.conv_block(inputs, training=training),
                _inputs,
            ])
        return self.relu(_sum)
    
    def build(self, input_shape):
        print(f"[DEBUG]: {self.name} -- input shape : {input_shape}")
        self.match_layer =   SkipLayer(
            output_filters=self.output_filters,
            name="Skip",
            trainable=self.trainable,
        )

        # Convolutional block
        self.conv_block = ConvBlockLayer(
            output_filters=self.output_filters,
            momentum=self.momentum,
            epsilon=self.epsilon,
            name="ConvBlock",
            trainable=self.trainable,
        )

        self.add = layers.Add(name="Add")
        self.relu= layers.ReLU(name="ReLu_identity",)  if self.use_last_relu else IdentityLayer(name="ReLu_identity")
        super().build(input_shape) 
        self.built = True
    
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        instance.match_layer =  None
        # Convolutional block
        instance.conv_block = None
        instance.add = None
        instance.relu= None
        return instance

@register_keras_serializable(package="lResiduals")    
class ResidualLayerIn(Layer):
    def __init__(
        self,
        output_filters: int,
        nblocks: int,
        momentum: float = 0.97,
        epsilon: float = 0.001,
        name: str = None,
        trainable: bool = True,
        use_last_relu: bool = False,
    ) -> None:
        super().__init__(name=name, trainable=trainable)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        self.nblocks = nblocks
        self.trainable = trainable

        self.residual_blocks = []

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_last_relu": self.use_last_relu,
                "nblocks": self.nblocks,
                "trainable": self.trainable
            },
        }
    
    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = 1.0*inputs
        for layer in self.residual_blocks:
            x = layer(x,training=training)
        return x
    
    def build(self, input_shape):
        print(f"[DEBUG]: {self.name} -- input shape : {input_shape}")
        self.residual_blocks.append(
            ResidualBlockIn(output_filters= self.output_filters,
                    name=f"{self.name}_IN",
                    use_last_relu = self.use_last_relu,
                    trainable=self.trainable)
                    )
        
        for k in range(self.nblocks-1):
            self.residual_blocks.append(
                ResidualBlock(output_filters= self.output_filters,
                    name=f"{self.name}_block{k}",
                    use_last_relu = self.use_last_relu,
                    trainable=self.trainable)
                )
        super().build(input_shape)
        self.built = True

    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        instance.residual_blocks = []
        return instance