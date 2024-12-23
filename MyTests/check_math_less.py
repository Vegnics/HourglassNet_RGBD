import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

T1 = tf.constant([[1.0,2.5,3.0,0.01],
                  [2.5,2.0,1.55,3.6],
                  [0.5,0.8,0.6,0.1],
                  [10.0,3.7,8.5,1.9],
                  [20.0,2.5,3.3,1.2]])

T2 = tf.expand_dims(tf.constant([2.0,1.6,0.6,3.0,3.2]),axis=1)
condition = tf.cast(tf.math.less(T1,T2*1.0),dtype=tf.float32)
print(condition)
print(tf.reduce_sum(condition).numpy())
precision = tf.float32
X, Y = tf.meshgrid(
            tf.range(
                start=0.0, limit=tf.cast(64, precision), delta=1.0, dtype=precision
            ),
            tf.range(
                start=0.0, limit=tf.cast(64, precision), delta=1.0, dtype=precision
            ),
        )
R1 = tf.math.square((X - 32.0)/1.2) + tf.math.square((Y - 32.0)/1.2)
Z = tf.exp(-0.5*R1)
plt.imshow(Z,cmap="jet")
plt.show()
print("NORM: ",tf.linalg.norm(Z,axis=[0,1]))

radius = 1
X, Y = tf.meshgrid(
            tf.range(
                start=-radius, limit=tf.cast(radius+1, tf.int32), delta=1, dtype=tf.int32
            ),
            tf.range(
                start=-radius, limit=tf.cast(radius+1, tf.int32), delta=1, dtype=tf.int32
            ),
        )
indices = tf.stack([X,Y],axis=-1)  
indices = tf.reshape(indices,(-1,2))
print(indices)