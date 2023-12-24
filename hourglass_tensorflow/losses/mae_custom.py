import tensorflow as tf
import keras.losses


class MAE_custom(keras.losses.Loss):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.AUTO, name="MAEcustom", *args, **kwargs
    ):
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        diff = y_true-y_pred
        mse = tf.reduce_mean(tf.math.abs(diff),axis=[2,3])
        mse = tf.reduce_sum(mse,axis=2)
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
