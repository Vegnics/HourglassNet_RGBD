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
        self.shared_conv = keras.Sequential([
            keras.layers.Conv2D(32, kernel_size=1, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(self.emb_len, strides=4, kernel_size=5, padding='same', activation='relu'),
            keras.layers.BatchNormalization()
        ])
        
        #Positional encoding
        self.pos_encoding = keras.layers.Embedding(input_dim=self.context_T, output_dim=self.emb_len,name="pos_encoding")
        
        #Last conv layer across frame descriptors
        self.last_conv = keras.layers.Conv2D(filters=16,strides=1,kernel_size=1, padding='same', activation='relu', name="last_conv")
        
        self.class_head = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(self.NActions,activation="softmax")
        ])
        
    def call(self,inputs,training=False):
        # Apply shared Conv2D to the sequence (spatial info -> frame descriptors)
        x = keras.layers.TimeDistributed(self.shared_conv)(inputs,training=training)  # Shape: (B, T, 16,16,64)
        
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
        outputs = self.class_head(x)
        return outputs