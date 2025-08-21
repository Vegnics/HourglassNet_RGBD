from os import name
import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.saving import register_keras_serializable
from keras import constraints

from hourglass_tensorflow.layers.sequential_layer import SequentialLayer
from hourglass_tensorflow.layers.skip import SkipLayer
from hourglass_tensorflow.layers.conv_block2 import ConvBlockLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer
from hourglass_tensorflow.layers.dummy_layers import zeroLayer,IdentityLayer, quasiConstantLayer
from hourglass_tensorflow.layers.attention_feature import FeatureAttentionMechanism
from hourglass_tensorflow.layers.attention_spatial4 import SpatialAttentionMechanism



@register_keras_serializable(package="lResidualsLoRa")
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
        feat_size: int = None,
        activate_lora: bool = None,
        include_metric: bool = False, 
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        self.attention_type = attentionType
        self.feat_size = feat_size
        self.activate_lora = activate_lora

        # Convolutional block
        self.attention_block = None
        if self.attention_type == "NoAM":
            self.attention_block = quasiConstantLayer(self.output_filters,name="AttentionBlock",value=1.0)
        
        elif self.attention_type == "SAM":
            self.attention_block = SpatialAttentionMechanism(
                name="AttentionBlock",
                filters = self.output_filters,
                kernel_size = 1,
                kernel_reg = False,
                trainable = self.trainable,
                feat_size = self.feat_size,
                include_metric=include_metric
            )
            
        elif self.attention_type == "FAM":
            self.attention_block = FeatureAttentionMechanism(
                name="AttentionBlock",
                filters = self.output_filters,
                kernel_size = 1,
                kernel_reg = False,
                trainable = self.trainable
            )
        else:
            raise Exception(f"[{self.name}]:INVALID ATTENTION MECHANISM")
            

        self.conv_block = ConvBlockLayer(
            output_filters=self.output_filters,
            momentum=self.momentum,
            epsilon=self.epsilon,
            name=f"ConvBlock",
            trainable=self.trainable,
            activate_lora = self.activate_lora
        )
        """
        self.lora_path = SequentialLayer(
                        [
                            layers.Conv2D(filters=16,
                                    kernel_size=(1,1),
                                    kernel_initializer="glorot_normal",
                                    name="lora_a",
                                    use_bias=False),
                            
                            layers.LayerNormalization(
                                axis=-1,
                                name="lora_ln"
                            ),
                            
                            layers.Conv2D(filters=32,
                                    kernel_size=(3,3),
                                    padding = "same",
                                    activation="gelu",
                                    kernel_initializer="glorot_uniform",
                                    name="lora_i0"),
                            
                            layers.Conv2D(filters=8,
                                    kernel_size=(3,3),
                                    padding = "same",
                                    activation="gelu",
                                    kernel_initializer="glorot_uniform",
                                    name="lora_i1"),

                            layers.Conv2D(filters=self.output_filters,
                                    kernel_size=(1,1),
                                    kernel_initializer="zeros",
                                    name="lora_b",
                                    use_bias=False)

                        ],
                        name="lora_path",
                        trainable=self.trainable
                    )
        """
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
                "attentionType": self.attention_type,
                "feat_size": self.feat_size,
                "activate_lora": self.activate_lora
            },
        }
    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        B = tf.shape(inputs)[0]        
        out_conv = self.conv_block(inputs, training=training)
        scores,scores_heads = self.attention_block(out_conv,training=training)
        #scores,scores_heads = self.attention_block(inputs,training=training)

        alpha = 0.8
        _sum = self.add(
            [  
                out_conv*(alpha*scores + (1-alpha)),
                inputs,
            ])
        
        #scores,scores_heads = self.attention_block(_sum,training=training)
        return self.relu(_sum),scores_heads #scores

    
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

@register_keras_serializable(package="lResidualsLoRa")
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
        feat_size: int = None,
        activate_lora: bool = None,
        include_metric = False,
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
        self.feat_size = feat_size
        self.activate_lora = activate_lora
        #self.residual_blocks = []

        self.residual_blocks = [ResidualBlock(output_filters= self.output_filters,
                                            name=f"{name}_block{k}",
                                            use_last_relu = self.use_last_relu,
                                            trainable=self.trainable,
                                            attentionType=self.attention_type,
                                            feat_size=self.feat_size,
                                            activate_lora=self.activate_lora,
                                            include_metric = include_metric) for k in range(self.nblocks)]
        for k,layer in enumerate(self.residual_blocks):
            self.__setattr__(f"residual_{k}", layer)
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_last_relu": self.use_last_relu,
                "nblocks": self.nblocks,
                "attentionType": self.attention_type,
                "feat_size":self.feat_size,
                "activate_lora": self.activate_lora
            },
        }
    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        #print(f"DEBUG_CALL: ResidualLayer '{self.name}' call method entered. Input shape: {inputs.shape}")
        x = 1.0*inputs
        scores = []
        for layer in self.residual_blocks:
            x,s = layer(x,training=training)
            scores.append(s)
        return x,tf.stack(scores,axis=-1)
    
    def build(self, input_shape):
        #print(f"[DEBUG]: {self.name} -- input shape : {input_shape}: CONFIG: {self.get_config()}")
        super().build(input_shape)
    
    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """
    
@register_keras_serializable(package="lResidualsLoRa")
class ResidualBlockIn(Layer):
    def __init__(
        self,
        output_filters: int,
        momentum: float = 0.98,
        epsilon: float = 0.001,
        name: str = None,
        trainable: bool = True,
        use_last_relu: bool = False,
        activate_lora: bool = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        self.activate_lora = activate_lora
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
            name=f"ConvBlock",
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
                "activate_lora": self.activate_lora
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

@register_keras_serializable(package="lResidualsLoRa")    
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
        activate_lora: bool = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_last_relu = use_last_relu
        self.nblocks = nblocks
        self.activate_lora = activate_lora

        self.residual_blocks = []
        self.residual_blocks.append(
            ResidualBlockIn(output_filters= self.output_filters,
                    name=f"{name}_IN",
                    use_last_relu = self.use_last_relu,
                    trainable=trainable,
                    activate_lora=self.activate_lora)
                    )
        
        for k in range(self.nblocks-1):
            print("DEBUG: Adding residual block ",k)
            self.residual_blocks.append(
                ResidualBlock(output_filters= self.output_filters,
                    name=f"{name}_block{k}",
                    use_last_relu = self.use_last_relu,
                    trainable=trainable,
                    attentionType="NoAM",
                    activate_lora=self.activate_lora)
                )
        for k,layer in enumerate(self.residual_blocks):
            self.__setattr__(f"residual_{k}", layer)
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_last_relu": self.use_last_relu,
                "nblocks": self.nblocks,
                "activate_lora":self.activate_lora
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