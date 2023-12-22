import sys
from typing import Any
import numpy as np
import tensorflow as tf
class Matros():
    def __init__(self,x) -> None:
        self.x = x
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.x**2

a = tf.constant([[[1,2],[3,2],[1,2]],
                 [[3,1],[7,8],[1,1]]],dtype=tf.dtypes.float32)
for i in range(3):
    b = tf.reduce_sum(a) if i==0 else b+tf.reduce_sum(a)
print(b.tolist())