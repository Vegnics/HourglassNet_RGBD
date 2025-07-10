from keras.layers import Layer
import keras
from keras.saving import register_keras_serializable 

@register_keras_serializable(package="lSequential")
class SequentialLayer(Layer):
    def __init__(self, layer_list, name=None,trainable=None,**kwargs):
        super().__init__(name=name,trainable=trainable,**kwargs)
        #self.layer_list = layer_list
        self.modelc = keras.Sequential(layer_list,name=name)
        for i, layer in enumerate(self.modelc.layers):
            self.__setattr__(f"layer_{i}", layer) 

    def call(self, inputs, training=False):
        return self.modelc(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        # Serialize inner layers
        config.update({
            "layer_list": [keras.layers.serialize(layer) for layer in self.modelc.layers]
        })
        return config

    @classmethod
    def from_config(cls, config):
        layer_list = [keras.layers.deserialize(l) for l in config.pop("layer_list")]
        return cls(layer_list=layer_list, **config)