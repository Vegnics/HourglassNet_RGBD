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

indexes = tf.constant([[0,1],[1,0]],dtype=tf.dtypes.int32)

b = tf.gather_nd(a,indexes)
#print(b)
x = tf.range(0,4,1)
y = tf.range(0,4,1)
X,Y = tf.meshgrid(x,y,indexing="ij")
X = tf.expand_dims(X,axis=2)
Y = tf.expand_dims(Y,axis=2)
indexes = tf.concat([X,Y],axis=2)
indexes = tf.reshape(indexes,shape=[-1,2])
indexes = tf.transpose(indexes,perm=[1,0])
angle = 15
angle = tf.constant(angle,dtype=tf.float32)*np.pi/180.0
center = tf.constant([2,2],dtype=tf.float32)
_angle = tf.expand_dims(angle,axis=0)
alpha = tf.math.cos(_angle) 
beta = tf.math.sin(_angle)
r1 = tf.expand_dims(tf.concat([alpha,beta],axis=0),axis=0)
r2 = tf.expand_dims(tf.concat([-beta,alpha],axis=0),axis=0)
R = tf.concat([r1,r2],axis=0)
a = tf.math.cos(angle)
b = tf.math.sin(angle)
t = tf.convert_to_tensor([[1-a,-b],[b,1-a]])@tf.reshape(center,shape=(-1,1))
_R = tf.convert_to_tensor([[a,b],[-b,a]])
#R = K@alphabeta
#R = tf.constant([[0,0],[-0.5,1]],dtype=tf.dtypes.float32)
indexes = tf.cast(indexes,tf.dtypes.float32)
r_indexes = _R@indexes + t
#check_mask = tf.cast(tf.math.less(r_indexes,0.0),dtype=tf.dtypes.int32)
check_mask = tf.math.greater_equal(r_indexes,0.0)
check_mask = check_mask[0,:]&check_mask[1,:]

_r_indexes = tf.floor(r_indexes)
_r_indexes = tf.transpose(_r_indexes,perm=[1,0])
_r_indexes = tf.boolean_mask(_r_indexes,check_mask)

r_indexes = tf.transpose(r_indexes,perm=[1,0])
r_indexes = tf.boolean_mask(r_indexes,check_mask)

print(r_indexes)
print(_r_indexes)
#print(check_mask)
#print(t)
#print(R)
#print(_R)
#print(a.shape)
#b = a * tf.reshape(tf.constant([1.0,2.0]),[-1,1,1])
#print(a.numpy())
#print(b.numpy())
#print(tf.reduce_mean(b,axis=2).numpy())
#for i in range(3):<
#    b = tf.reduce_sum(a) if i==0 else b+tf.reduce_sum(a)
#W = tf.constant([1.0,2.0,3.0,4.0,4.0,5.0])
#W = tf.reshape(W,[1,-1,1,1])
#print(W.shape)