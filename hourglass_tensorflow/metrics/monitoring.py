import tensorflow as tf
from keras.metrics import Metric
from typing import Tuple
from keras.utils import register_keras_serializable

@register_keras_serializable(package="mymetrics")
class MagnitudeLossMonitor(Metric):

    def __init__(
        self,
        name=None,
        num_1joints: int = 14, 
        dtype=None,
        **kwargs
    ) -> None:
        super().__init__(name=name, dtype=tf.float32, **kwargs)
        self.monitor_magnitude = self.add_weight(
            name="magnitude_loss_b", initializer="zeros"
        )
        self.batch_count = self.add_weight(
            name="batch_count", initializer="zeros")
        
        self.n1joints = num_1joints

    def _internal_update(self, y_true, y_pred):

        mask_joints_1 = tf.where(tf.reduce_max(y_true[:,-1,:,:,0:self.n1joints],axis=[1,2])>0.0001,1.0,0.0) #NC
        joint_count_1 = tf.reduce_sum(mask_joints_1,axis=1)#N

        # GT and pred sums
        mag_true =  tf.reduce_sum(y_true[:,:,:,:,0:self.n1joints],axis=[2,3]) #N,S,C
        mag_pred = tf.reduce_sum(y_pred[:,:,:,:,0:self.n1joints],axis=[2,3]) #N,S,C

        # Cumulative heatmap regression loss
        cum_diff = tf.math.abs(mag_true-mag_pred) #N,S,C
        cum_diff = tf.reduce_mean(cum_diff,axis=1) #N,C
        cum_loss = (tf.reduce_sum(cum_diff*tf.cast(mask_joints_1,dtype=tf.float32),axis=-1))/(tf.cast(joint_count_1,dtype=tf.float32)+0.001)
        cum_loss = tf.reduce_mean(cum_loss)

        self.monitor_magnitude.assign_add(cum_loss)
        self.batch_count.assign_add(1.0)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        return self._internal_update(y_true, y_pred)

    def result(self, *args, **kwargs):
        return tf.math.divide_no_nan(self.monitor_magnitude,self.batch_count)

    def reset_state(self) -> None:
        self.monitor_magnitude.assign(0.0)
        self.batch_count.assign(0.0)