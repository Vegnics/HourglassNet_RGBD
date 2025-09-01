import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.saving import register_keras_serializable

@register_keras_serializable(package="lDummy")
class IdentityLayer(Layer):
    def __init__(
        self,
        name: str = None,
        trainable: bool = False,
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)

    def call(self,input) -> tf.Tensor:
        return input
    
    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def get_config(self):
        return super().get_config() 
    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """
    
@register_keras_serializable(package="lDummy")
class zeroLayer(Layer):
    def __init__(
        self,
        output_channels: int = None,
        name: str = None,
        trainable=False,
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store Config
        self.output_channels = output_channels 
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_channels": self.output_channels,
            },
        }
    def call(self,inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        return tf.zeros([batch_size, height, width, self.output_channels], dtype=inputs.dtype)

    def build(self,input_shape):
        super().build(input_shape)
        self.built = True


@register_keras_serializable(package="lDummy")
class  ConstantScoreLayer(Layer):
    def __init__(
        self,
        output_channels: int = None,
        name: str = None,
        trainable=False,
        value: float = None,
        stack_num: int = 4,
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store Config
        self.output_channels = output_channels
        self.value_out = value 
        self.stack_num = stack_num
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_channels": self.output_channels,
                "value":self.value_out,
                "stack_num": self.stack_num
            },
        }
    def call(self, inputs, y_true=None, training=False):
    #def call(self,inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        outs = tf.ones(shape=[batch_size, height, width, self.output_channels], dtype=inputs.dtype)
        stack_out = tf.ones(shape=[batch_size, height, width, self.stack_num], dtype=inputs.dtype)
        #outs = tf.random.uniform(shape=[batch_size, height, width, self.output_channels],minval=self.value_out,maxval=self.value_out+0.005, dtype=inputs.dtype)
        #outs = tf.random.normal(shape=[batch_size, height, width, self.output_channels],mean=self.value_out,stddev=0.2, dtype=inputs.dtype)
        return outs,stack_out #tf.nn.sigmoid(outs) #self.value_out*outs #tf.ones([batch_size, height, width, self.output_channels], dtype=inputs.dtype)

    def build(self,input_shape):
        super().build(input_shape)
        self.built = True

    """
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance
    """