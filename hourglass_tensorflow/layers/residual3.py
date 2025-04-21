from os import name
import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.saving import register_keras_serializable
from keras import constraints

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
        feat_size: int = None, 
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
        self.alpha_mask = tf.constant(1.0,dtype=tf.float32)
        #self.ff_alpha = lambda x:x
        # Convolutional block
        self.attention_block = None
        if self.attention_type == "NoAM":
            self.attention_block = zeroLayer(self.output_filters,name="AttentionBlock")
            self.alpha = tf.constant(-3.0,dtype=tf.float32)
            self.alpha_mask = 0.0*self.alpha_mask
        
        elif self.attention_type == "SAM":
            self.attention_block = SpatialAttentionMechanism(
                name="AttentionBlock",
                filters = self.output_filters,
                kernel_size = 1,
                kernel_reg = False,
                trainable = self.trainable,
                feat_size = self.feat_size
            )
            self.alpha = self.add_weight(
                shape=[1,],
                name="att_alpha",
                initializer=tf.constant_initializer(-2.8),
                constraint=constraints.max_norm(3.0),
                trainable=self.trainable,
            )
        elif self.attention_type == "FAM":
            self.attention_block = FeatureAttentionMechanism(
                name="AttentionBlock",
                filters = self.output_filters,
                kernel_size = 1,
                kernel_reg = False,
                trainable = self.trainable
            )
            self.alpha = self.add_weight(
                shape=[1,],
                name="att_alpha",
                initializer=tf.constant_initializer(-2.8),
                constraint=constraints.max_norm(3.0),
                trainable=self.trainable,
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
                "attentionType": self.attention_type,
                "feat_size": self.feat_size
            },
        }
    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        #B = tf.shape(inputs)[0]
        scores = self.attention_block(inputs,training=training)
        #entropy = -1.0*tf.reduce_sum(scores*tf.math.log(tf.math.maximum(scores,1e-5)),axis=[1,2,3])
        #numel = tf.cast(tf.reduce_prod(tf.shape(scores)[1:]), tf.float32)
        #entropy = entropy / numel
        #r_alpha = self.alpha*tf.ones_like(entropy)
        #in_ffalpha = tf.stack([entropy,r_alpha],axis=1)
        alpha = self.alpha_mask*self.alpha
        _sum = self.add(
            [
                self.conv_block(inputs, training=training),
                inputs*(alpha*scores+(1.0-alpha)),
            ])
        return self.relu(_sum)
    
    def build(self, input_shape):
        #print(f"[DEBUG]: {self.name} -- input shape : {input_shape}: CONFIG: {self.get_config()}")
        #self.ff_alpha = layers.Dense(1,activation="sigmoid",
        #                             kernel_initializer="zeros",
        #                             name="ff_alpha",
        #                             bias_initializer=tf.constant_initializer(-2.5))
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
        feat_size: int = None,
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
        #self.residual_blocks = []

        self.residual_blocks = [ResidualBlock(output_filters= self.output_filters,
                                            name=f"{name}_block{k}",
                                            use_last_relu = self.use_last_relu,
                                            trainable=trainable,
                                            attentionType=self.attention_type,
                                            feat_size=self.feat_size) for k in range(self.nblocks)]
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
                "feat_size":self.feat_size
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