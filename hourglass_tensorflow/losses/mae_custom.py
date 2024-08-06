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
        #L2 = tf.abs(tf.reduce_sum(y_true,axis=[2,3])-tf.reduce_sum(y_pred,axis=[2,3]))/(64.0*64.0)
        #L3 = tf.square(y_true) - y_true*y_pred
        #L3 = tf.abs(L3)
        #W0 = tf.constant([0.3,0.3,0.5],dtype=tf.dtypes.float32)
        #W0 = tf.reshape(W0,[1,-1,1,1,1])
        #L3 = tf.reduce_sum(W0*L3,axis=1)
        #L3 = tf.reduce_mean(L3,axis=[1,2])
        #L3 = tf.reduce_sum(L3,axis=1)

        W = tf.constant([0.33,0.33,0.33],dtype=tf.dtypes.float32)
        W = tf.reshape(W,[1,-1,1,1,1])
        #Rmax = tf.sqrt(tf.constant(2.0,dtype=tf.float32))*tf.constant(64.0,dtype=tf.float32)
        #cy_true = tf.exp(-0.5*tf.square(Rmax*tf.cast(y_true,dtype=tf.dtypes.float32)/255.0))
        #cy_true = tf.cast(y_true,dtype=tf.dtypes.float32)/255.0
        #dist1 = tf.abs(tf.math.square(1.0+(cy_true-y_pred))-1.0)#NSHWC
        
        
        #sdiff = (tf.abs(y_true-y_pred))
        # Apply the logarithm first 
        # Ytrue-> Positive [eps,1]
        # Ypred-> +-
        #sdiff = -1.0*tf.math.log(y_true+0.00001) + tf.math.log(y_pred+0.00001)

        normtrue = tf.norm(y_true,axis=[2,3])
        normpred = tf.norm(y_pred,axis=[2,3])
        normtrue = tf.reshape(normtrue,[-1,3,1,1,16])
        normpred = tf.reshape(normpred,[-1,3,1,1,16])

        #normtrue = tf.reshape(normtrue,[-1,3,1,1,14])
        #normpred = tf.reshape(normpred,[-1,3,1,1,14])
        
        #Scaled and shifted square loss function 
        #sdiff = tf.math.abs(tf.math.square(10.0*y_true+1.0) - tf.math.square(10.0*y_pred+1.0))
        #dist1 = sdiff*W b
        #dist1 = tf.reduce_sum(dist1,axis=1) #NSHW
        #dist1 = tf.reduce_mean(dist1,axis=3)
        #dist1 = tf.reduce_mean(dist1,axis=[1,2])

        #Normalized cross correlation loss function 
        eps = 0.001
        sdiff = tf.reduce_sum(y_true*y_pred/(normpred*normtrue+eps),axis=[2,3]) #NSC
        #dist1 = -1.0*tf.math.sigmoid(0.5*tf.reduce_mean(sdiff,axis=2)-2.0)
        dist1 = -1.0*tf.math.exp(tf.reduce_mean(sdiff,axis=2))
        dist1 = tf.reduce_mean(dist1,axis=1)
        
        #Log Abs loss
        #_normtrue = tf.reshape(normtrue,[-1,3,16])
        #_normpred = tf.reshape(normpred,[-1,3,16])
        
        #sdiff2 = tf.abs(y_true-y_pred)#+eps
        #sdiff2 = tf.reduce_mean(sdiff2,axis=[2,3])
        #dist2 = tf.reduce_mean(sdiff2,axis=1)
        #dist2 = tf.reduce_mean(dist2,axis=1)
        #dist2  = -1.0*tf.math.exp(tf.math.divide_no_nan(tf.constant(1.0),dist2+eps))



        #dist1 = (0.1*tf.math.pow(y_true,1/32) + 1.0) * sdiff
        #dist1 = dist1
        #dist2 = tf.math.abs(cy_true-y_pred)*W
        #dist1 = dist1 + dist2
        #dist = tf.debugging.check_numerics(dist, message='Checking DIST')
        """
        dist = tf.reduce_mean(dist,axis=1) #NHWC
        dist= tf.reduce_sum(dist,axis=[1,2])/64.0 #NC
        dist = tf.reduce_sum(dist,axis=1)# N
        """
        
        #dist1 = tf.reduce_sum(dist1,axis=4) #NSHW
        #dist1 = tf.reduce_mean(dist1,axis=3)
        #dist1 = tf.reduce_mean(dist1,axis=[1,2]) #NS
        #dist1 = tf.reduce_mean(dist1,axis=[1,2]) 
        #dist1= tf.sqrt(tf.reduce_mean(dist1,axis=[1,2])) #N
        #dist1 = tf.reduce_sum(dist1,axis=1) #N
        #dist2 = tf.reduce_sum(dist2,axis=1) #NHWC
        #dist2 = tf.reduce_sum(dist2,axis=3) #NHW
        #dist2= tf.reduce_mean(dist2,axis=[1,2]) #N

        #dist = 0.8*dist1 + 0.2*dist2
        return dist1#+0.1*dist2