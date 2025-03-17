import tensorflow as tf
from keras import layers
from keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="lDummy")
class IdentityLayer(Layer):
    def __init__(
        self,
        name: str = None
    ) -> None:
        super().__init__(name=name, trainable=False)

    def call(self,input) -> tf.Tensor:
        return input
    
    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def get_config(self):
         return {
            **super().get_config(),
            **{
            },
        }
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable(package="lDummy")
class zeroLayer(Layer):
    def __init__(
        self,
        output_channels: int = None,
        name: str = None,
    ) -> None:
        super().__init__(name=name, trainable=False)
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
        x = tf.reduce_sum(0.0*inputs,axis=3)
        x = tf.expand_dims(x,axis=3)
        return x*tf.zeros(shape=(1,1,1,self.output_channels))
    def build(self,input_shape):
        super().build(input_shape)
        self.built = True

    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance