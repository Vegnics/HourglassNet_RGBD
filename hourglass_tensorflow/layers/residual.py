import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.saving import register_keras_serializable

from hourglass_tensorflow.layers.skip import SkipLayer
from hourglass_tensorflow.layers.conv_block import ConvBlockLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer
from hourglass_tensorflow.layers.dummy_layers import IdentityLayer
from hourglass_tensorflow.layers.attention_feature import FeatureAttentionMechanism
from hourglass_tensorflow.layers.attention_spatial import SpatialAttentionMechanism
from hourglass_tensorflow.layers.dummy_layers import zeroLayer

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
        attentionType: str = None, 
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        self.attention_type = attentionType
        # Convolutional block

        self.attention_block = None
        if self.attention_type == "NoAM":
            self.attention_block = zeroLayer(self.output_filters,name="AttentionBlock")
        
        elif self.attention_type == "SAM":
            self.attention_block = SpatialAttentionMechanism(
                name="AttentionBlock",
                filters = self.output_filters,
                kernel_size = 1,
                kernel_reg = False,
                trainable = True
            )
        elif self.attention_type == "FAM":
            FeatureAttentionMechanism(
                name="AttentionBlock",
                filters = self.output_filters,
                kernel_size = 1,
                kernel_reg = False,
                trainable = True
            )
        else:
            raise Exception(f"[{self.name}]:INVALID ATTENTION MECHANISM")
            

        self.conv_block = ConvBlockLayer(
            output_filters=self.output_filters,
            momentum=self.momentum,
            epsilon=self.epsilon,
            name="ConvBlock",
            trainable=trainable,
        )
        self.add = layers.Add(name="Add")
        self.relu= layers.ReLU(name="ReLu_identity",)  if self.use_last_relu else IdentityLayer(name="ReLu_identity")
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_last_relu": self.use_last_relu,
                "attentionType": self.attention_type
            },
        }
    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        scores = self.attention_block(inputs)
        _sum = self.add(
            [
                self.conv_block(inputs, training=training),
                inputs*scores,
                inputs,
            ])
        return self.relu(_sum)
    
    def build(self, input_shape):
        #print(f"[DEBUG]: {self.name} -- input shape : {input_shape}: CONFIG: {self.get_config()}")
        super().build(input_shape)
        self.built = True

    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """

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
        attentionType: str = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        self.nblocks = nblocks
        self.attention_type = attentionType
        #self.residual_blocks = []

        self.residual_blocks = [ResidualBlock(output_filters= self.output_filters,
                                            name=f"{name}_block{k}",
                                            use_last_relu = self.use_last_relu,
                                            trainable=trainable,
                                            attentionType=self.attention_type) for k in range(self.nblocks)]

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_last_relu": self.use_last_relu,
                "nblocks": self.nblocks,
                "attentionType": self.attention_type
            },
        }
    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        x = 1.0*inputs
        for layer in self.residual_blocks:
            x = layer(x,training=training)
        return x
    
    def build(self, input_shape):
        #print(f"[DEBUG]: {self.name} -- input shape : {input_shape}: CONFIG: {self.get_config()}")
        super().build(input_shape)
    
    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """
    
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
        **kwargs,
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        # Input layer (used just to match the dimensionality)
        
        #self.match_layer =  None
        #self.conv_block = None
        #self.add = None
        #self.relu= None

        self.match_layer =   SkipLayer(
            output_filters=self.output_filters,
            name="Skip",
            trainable=trainable,
        )

        # Convolutional block
        self.conv_block = ConvBlockLayer(
            output_filters=self.output_filters,
            momentum=self.momentum,
            epsilon=self.epsilon,
            name="ConvBlock",
            trainable=trainable,
        )

        self.add = layers.Add(name="Add")
        self.relu= layers.ReLU(name="ReLu_identity",)  if self.use_last_relu else IdentityLayer(name="ReLu_identity")

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

    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        _inputs = self.match_layer(inputs,training=training)
        _sum = self.add(
            [
                self.conv_block(inputs, training=training),
                _inputs,
            ])
        return self.relu(_sum)
    
    def build(self, input_shape):
        #print(f"[DEBUG]: {self.name} -- input shape : {input_shape}: CONFIG: {self.get_config()}")
        super().build(input_shape) 
    
    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """

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
        **kwargs,
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        self.nblocks = nblocks

        self.residual_blocks = []
        self.residual_blocks.append(
            ResidualBlockIn(output_filters= self.output_filters,
                    name=f"{name}_IN",
                    use_last_relu = self.use_last_relu,
                    trainable=trainable)
                    )
        
        for k in range(self.nblocks-1):
            self.residual_blocks.append(
                ResidualBlock(output_filters= self.output_filters,
                    name=f"{name}_block{k}",
                    use_last_relu = self.use_last_relu,
                    trainable=trainable,
                    attentionType="NoAM")
                )

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_last_relu": self.use_last_relu,
                "nblocks": self.nblocks
            },
        }
    
    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        x = 1.0*inputs
        for layer in self.residual_blocks:
            x = layer(x,training=training)
        return x
    
    def build(self, input_shape):
        #print(f"[DEBUG]: {self.name} -- input shape : {input_shape}: CONFIG: {self.get_config()}")
        super().build(input_shape)

    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """