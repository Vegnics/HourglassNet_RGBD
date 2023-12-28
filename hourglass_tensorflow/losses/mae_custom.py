import tensorflow as tf
import keras.losses


class MAE_custom(keras.losses.Loss):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.AUTO, name="MAEcustom", *args, **kwargs
    ):
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        #01234
        #NSHWC
        #NHWC
        #W = tf.constant([2.0,2.5,3.0,3.0,3.0,8.0],dtype=tf.dtypes.float32)
        #W = tf.reshape(W,[1,-1,1,1,1])
        dist = tf.math.abs(y_true-y_pred)#*W
        #cos = tf.math.greater_equal(y_true*y_pred,0.0)
        mse = tf.reduce_sum(dist,axis=1)
        mse = tf.reduce_mean(mse,axis=[1,2])
        #mse = tf.reduce_sum(mse,axis=[1,2])
        mse = tf.reduce_mean(mse,axis=1)
        #sqr = tf.reduce_sum(tf.math.abs(diff),axis=1)
        #mae = tf.reduce_sum(sqr,axis=3)
        #mae = tf.reduce_mean(mae,axis=[1,2])
        print(">>>>MAE SHAPE: ",mse.shape)
        return mse
        #return tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=y_pred,
        #    labels=y_true,
        #    name="nn.sigmoid_cross_entropy_with_logits",
        #)
