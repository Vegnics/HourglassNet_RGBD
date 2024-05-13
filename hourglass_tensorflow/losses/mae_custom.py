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
        #NSHW
        #NHW
        #W = tf.constant([0.05,0.1,0.15,0.18,0.22,0.3],dtype=tf.dtypes.float32)
        #W = tf.constant([0.05,0.15,0.2,0.25,0.3,0.25],dtype=tf.dtypes.float32)
        #W = tf.constant([3.0,2.0,3.0],dtype=tf.dtypes.float32)
        #W = tf.constant([0.48,0.52],dtype=tf.dtypes.float32)
        #W = tf.constant([0.32,0.33,0.35],dtype=tf.dtypes.float32)
        W = tf.constant([0.33,0.33,0.33],dtype=tf.dtypes.float32)
        W = tf.reshape(W,[1,-1,1,1,1])
        #Rmax = tf.sqrt(tf.constant(2.0,dtype=tf.float32))*tf.constant(64.0,dtype=tf.float32)
        #cy_true = tf.exp(-0.5*tf.square(Rmax*tf.cast(y_true,dtype=tf.dtypes.float32)/255.0))
        #cy_true = tf.cast(y_true,dtype=tf.dtypes.float32)/255.0
        #dist1 = tf.abs(tf.math.square(1.0+(cy_true-y_pred))-1.0)#NSHWC
        sdiff = (tf.abs(y_true-y_pred))
        dist1 = (0.1*tf.math.pow(y_true,1/32) + 1.0) * sdiff
        dist1 = dist1*W
        #dist1 = dist1
        #dist2 = tf.math.abs(cy_true-y_pred)*W
        #dist1 = dist1 + dist2
        #dist = tf.debugging.check_numerics(dist, message='Checking DIST')
        """
        dist = tf.reduce_mean(dist,axis=1) #NHWC
        dist= tf.reduce_sum(dist,axis=[1,2])/64.0 #NC
        dist = tf.reduce_sum(dist,axis=1)# N
        """
        dist1 = tf.reduce_sum(dist1,axis=1) #NHWC
        #dist1 = tf.reduce_sum(dist1,axis=4) #NSHW
        dist1 = tf.reduce_mean(dist1,axis=3)
        dist1 = tf.reduce_mean(dist1,axis=[1,2]) #NS
        #dist1 = tf.reduce_mean(dist1,axis=[1,2]) 
        #dist1= tf.sqrt(tf.reduce_mean(dist1,axis=[1,2])) #N
        #dist1 = tf.reduce_sum(dist1,axis=1) #N
        #dist2 = tf.reduce_sum(dist2,axis=1) #NHWC
        #dist2 = tf.reduce_sum(dist2,axis=3) #NHW
        #dist2= tf.reduce_mean(dist2,axis=[1,2]) #N

        #dist = 0.8*dist1 + 0.2*dist2
        return dist1