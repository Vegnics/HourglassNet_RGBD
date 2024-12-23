import tensorflow as tf
import keras.losses


class MAE_custom(keras.losses.Loss):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.AUTO, name="MAEcustom", *args, **kwargs
    ):
        super().__init__(reduction, name)
        self.stages = 10
        self.njoints = 10
        for key, value in kwargs.items():
            if key == "nstages":
                self.stages = int(value)
            elif key == "njoints":
                self.njoints = int(value)
            
    def call(self, y_true, y_pred):
        #01234
        #NSHWC
        #NSHW
        #NHW
        S = self.stages
        C = self.njoints
        """
        _y_true = tf.reshape(y_true,shape=[-1,S,64,64,C])
        _y_pred = tf.reshape(y_pred,shape=[-1,S,64,64,C])
        _normtrue = tf.norm(y_true,axis=[2,3])
        _normpred = tf.norm(y_pred,axis=[2,3])
        normtrue = tf.reshape(_normtrue,[-1,S,1,1,C])
        normpred = tf.reshape(_normpred,[-1,S,1,1,C])
        wXC1 = tf.ones(shape=[1,S-1])
        wXC0 = tf.ones(shape=[1,1])
        WXC = tf.concat([wXC1,wXC0],axis=1)
        """
        #Normalized cross correlation loss function
        
        
        #_WC1 = [0.0]*16
        #_WC1 = _WC1 + [1.0]*15
        #WC1 = tf.constant([_WC1])
        #WC1 = tf.reshape(WC1,shape=[1,1,C])

        #_WC2 = [1.0]*16
        #_WC2 = _WC2 + [0.0]*15
        #WC2 = tf.constant([_WC2])
        #WC2 = tf.reshape(WC2,shape=[1,1,C])
        Ws = tf.reshape(tf.constant([0.25,0.35,0.4]),[1,3])
        """ 
        eps = tf.reshape(tf.constant(0.000001),shape=[-1,1,1,1,1])
        _normpred  = tf.reduce_sum(y_pred,axis=[2,3])
        normpred = tf.reshape(_normpred,[-1,S,1,1,C])+eps
        #sdiff = 1.0-tf.reduce_sum((y_true)*(y_pred)/(normpred*normtrue+eps),axis=[2,3]) #NSC
        sdiff = -tf.reduce_sum(y_true*tf.math.log(y_pred/normpred+eps),axis=[2,3])
        #sdiff = tf.math.exp(-0.8*sdiff)
        #dist1 = -10.0*sdiff
        
        #Wc = tf.reshape(tf.convert_to_tensor([1]*14+[0.5]*12),[1,1,26])/(14+6)
        #dist1 = tf.reduce_sum(sdiff*Wc,axis=2)
        
        dist1 = tf.reduce_mean(sdiff,axis=2)
        dist1 = tf.reduce_sum(dist1*Ws,axis=1)
        dist1 = tf.reduce_mean(dist1)
        """


        #"""
        wEMAE0 = tf.zeros(shape=[1,S-1])
        wEMAE1 = tf.ones(shape=[1,1])
        WEMAE = tf.concat([wEMAE0,wEMAE1],axis=1)
        #Exponential Abs Difference
        #ndiff = (1.0+32.0*tf.math.sqrt(_y_true))*tf.abs(_y_true-_y_pred)
        ndiff = tf.math.square(y_true-y_pred)
        #ndiff = tf.sqrt(tf.reduce_mean(ndiff+0.000000001,axis=[2,3])) #NSHW
        ndiff = tf.reduce_mean(ndiff+0.00000001,axis=[2,3]) #NSHW
         #+0.00001
        #sumytrue = tf.reduce_mean(_y_true,axis=[2,3]) #NSC
        #sumypred = tf.reduce_mean(_y_pred,axis=[2,3]) #NSC
        #sndiff = tf.abs(sumytrue-sumypred)
        #dist2 = 10.0*tf.math.exp(0.9*sndiff)
        #dist2 = -1.0*tf.math.exp(1.0/(tf.reduce_mean(ndiff,axis=[2,3]))) #NSC
        
        #dist2 = tf.math.log(tf.reduce_mean(ndiff,axis=[2,3])) #NSC
        Wc = tf.reshape(tf.convert_to_tensor([1]*14+[0.5]*12),[1,1,26])/(14+6)
        dist2 = tf.reduce_sum(ndiff*Wc,axis=2) #NS
        #dist2 = tf.sqrt(tf.reduce_mean(ndiff[:,:,:,:,0:16],axis=[2,3])) #NSC
        #dist3 = tf.sqrt(tf.reduce_mean(ndiff[:,:,:,:,16:31],axis=[2,3])) #NSC
        
        
        #dist2 = tf.reduce_sum(dist2/16,axis=2)# ^2.0
        dist2 = tf.reduce_mean(dist2,axis=1) #N
        dist2 = tf.reduce_mean(dist2)

        #dist3 = tf.reduce_sum(dist3/15,axis=2)# ^2.0
        #dist3 = tf.reduce_mean(dist3,axis=1)
        #dist3 = tf.reduce_mean(dist3)
        
        
        #dist2 = -10.0*dist2
        #dist = dist1 + 0.15*dist2
        #"""

        return dist2