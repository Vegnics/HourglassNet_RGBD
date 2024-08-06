import tensorflow as tf
import numpy as np

T1 = tf.constant([[1.0,2.5,3.0,0.01],
                  [2.5,2.0,1.55,3.6],
                  [0.5,0.8,0.6,0.1],
                  [10.0,3.7,8.5,1.9],
                  [20.0,2.5,3.3,1.2]])

T2 = tf.expand_dims(tf.constant([2.0,1.6,0.6,3.0,3.2]),axis=1)
condition = tf.cast(tf.math.less(T1,T2*1.0),dtype=tf.float32)
print(condition)
print(tf.reduce_sum(condition).numpy())