from hourglass_tensorflow.layers.sequential_layer import SequentialLayer
from keras.layers import Layer
import keras
import tensorflow as tf

class ActionRecognitionLayer(Layer):
    def __init__(self,name,
                 context_T,
                 NJoints,
                 NActions,
                 emb_len,
                 trainable,**kwargs):
        self.NJoints = NJoints
        self.NActions = NActions
        self.context_T = context_T
        self.emb_len = emb_len
        super().__init__(name=name,trainable=trainable,**kwargs)
        
        # Shared Conv2D
        self.shared_conv = SequentialLayer([
                keras.layers.Conv2D(32, kernel_size=1, padding='same', activation='relu'),
                keras.layers.BatchNormalization(name="BN_AR1"),
                keras.layers.Conv2D(self.emb_len, strides=4, kernel_size=5, padding='same', activation='relu'),
                keras.layers.BatchNormalization(name="BN_AR2")
            ],
            name="sequential_shared_conv")
        # time distributed layer (per-frame processing)
        self.t_distributed = keras.layers.TimeDistributed(self.shared_conv,name="time_distributed")
        
        #Positional encoding
        self.pos_encoding = keras.layers.Embedding(input_dim=self.context_T, output_dim=self.emb_len,name="pos_encoding")
        
        #Last conv layer across frame descriptors
        self.last_conv = keras.layers.Conv2D(filters=16,strides=1,kernel_size=1, padding='same', activation='relu', name="last_conv")
        
        self.class_head = SequentialLayer([
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.1,name="class_dropout"),
                keras.layers.Dense(self.NActions,activation="softmax")
            ],
            name="sequential_class_head")
        
    def call(self,inputs,training=False):
        # Apply shared Conv2D to the sequence (spatial info -> frame descriptors)
        x = self.t_distributed(inputs,training=training)  # Shape: (B, T, 16,16,64)
        
        # Add positional Encoding
        pos_encoding = self.pos_encoding(tf.range(self.context_T))
        pos_encoding = tf.reshape(pos_encoding, (1,self.context_T, 1, 1, self.emb_len))
        x = x + pos_encoding 

        # extract per-frame descriptors via avg pooling
        x = tf.transpose(tf.reduce_mean(x,axis=-1),perm=[0,2,3,1])# Shape: (B,16,16,T)

        # Further temporal processing
        x = self.last_conv(x) # (B,16,16,16)
        x = tf.reshape(x,shape=(-1,16*16*16))

        # Classification head
        outputs = self.class_head(x,training=training)
        return outputs #tf.expand_dims(outputs,axis=1)
    
    def get_config(self):
        return {
            "NJoints":  self.NJoints,
            "NActions": self.NActions,
            "context_T": self.context_T,
            "emb_len": self.emb_len,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)