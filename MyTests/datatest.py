import sys
from typing import Any
import numpy as np
import tensorflow as tf
class Matros():
    def __init__(self,x) -> None:
        self.x = x
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.x**2

a = tf.constant([[[1,2,1],[3,2,1],[1,2,3]],
                 [[3,1,2],[2,1,2],[1,1,2]]],dtype=tf.dtypes.float32)
print(a.shape)
b = a * tf.reshape(tf.constant([1.0,2.0]),[-1,1,1])
print(a.numpy())
print(b.numpy())
print(tf.reduce_mean(b,axis=2).numpy())
#for i in range(3):<
#    b = tf.reduce_sum(a) if i==0 else b+tf.reduce_sum(a)
#W = tf.constant([1.0,2.0,3.0,4.0,4.0,5.0])
#W = tf.reshape(W,[1,-1,1,1])
#print(W.shape)