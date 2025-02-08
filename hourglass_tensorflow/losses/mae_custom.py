import tensorflow as tf
import keras.losses
from hourglass_tensorflow.utils.tf import tf_batch_matrix_softargmax,tf_batch_multistage_matrix_softargmax_loss


class MAE_custom(keras.losses.Loss):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.AUTO, name="MAEcustom", *args, **kwargs
    ):
        super().__init__(reduction=None, name=name)
        self.stages = 3
        self.n1joints = 14
        self.n2joints = 12
        self.use2joints = True
        for key, value in kwargs.items():
            if key == "nstages":
                self.stages = int(value)
            elif key == "n1joints":
                self.n1joints = int(value)
            elif key == "n2joints":
                self.n2joints = int(value)
            elif key == "use2joints":
                self.use2joints = int(value)
        channel_mask = tf.reshape(tf.convert_to_tensor([1]*self.n1joints+[self.use2joints]*self.n2joints),shape=(1,1,1,1,self.n1joints+self.n2joints))
        self.channel_mask = tf.cast(channel_mask, dtype = tf.float32)
        self.Nchannels = tf.reduce_sum(self.channel_mask)
    def call(self, y_true, y_pred):
        #01234
        #NSHWC
        #NSHW
        #NHW
        S = self.stages
        #C = self.njoints
        #_y_true = y_true*self.channel_mask #NSHWC
        #_y_pred = y_pred*self.channel_mask #NSHWC
        
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
        
        Ws = tf.reshape(tf.constant([0.31,0.33,0.36]),[1,3])
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
        gt_coords = tf_batch_multistage_matrix_softargmax_loss(y_true[:,:,:,:,0:self.n1joints])
        pred_coords = tf_batch_multistage_matrix_softargmax_loss(y_pred[:,:,:,:,0:self.n1joints])
        loss_coords = tf.math.square(gt_coords-pred_coords)+0.0001
        loss_coords = tf.math.sqrt(tf.reduce_mean(loss_coords))

        wEMAE0 = tf.zeros(shape=[1,S-1])
        wEMAE1 = tf.ones(shape=[1,1])
        WEMAE = tf.concat([wEMAE0,wEMAE1],axis=1)
        #Exponential Abs Difference
        #ndiff = (1.0+32.0*tf.math.sqrt(_y_true))*tf.abs(_y_true-_y_pred)
        ndiff = tf.math.square(y_true-y_pred)
        #01234
        #NSHWC
        #ndiff = tf.sqrt(tf.reduce_mean(ndiff+0.000000001,axis=[2,3])) #NSHW
        ndiff = tf.reduce_mean(ndiff+0.00000001,axis=[2,3]) #NSHW  0.00000001
         #+0.00000001
        #sumytrue = tf.reduce_mean(_y_true,axis=[2,3]) #NSC
        #sumypred = tf.reduce_mean(_y_pred,axis=[2,3]) #NSC
        #sndiff = tf.abs(sumytrue-sumypred)
        #dist2 = 10.0*tf.math.exp(0.9*sndiff)
        #dist2 = -1.0*tf.math.exp(1.0/(tf.reduce_mean(ndiff,axis=[2,3]))) #NSC
        
        #dist2 = tf.math.log(tf.reduce_mean(ndiff,axis=[2,3])) #NSC
        #Wc = tf.reshape(tf.convert_to_tensor([1]*self.n1joints+[0.7]*self.n2joints),[1,1,self.n1joints+self.n2joints])*self.channel_mask[:,:,0,0,:]
        #SumWc = tf.reduce_sum(Wc)
        #Wc = Wc/SumWc
        
        loss_1jnt = tf.reduce_mean(ndiff[:,:,0:self.n1joints],axis=2) #NS
        loss_2jnt = tf.reduce_mean(ndiff[:,:,self.n1joints:self.n2joints],axis=2)#NS

        dist2 = 1.5*tf.reduce_mean(loss_1jnt)+0.5*loss_coords+0.3*tf.cast(self.use2joints,dtype=tf.float32)*tf.reduce_mean(loss_2jnt)
        #dist2 = tf.reduce_sum(ndiff*Wc,axis=2) #NS
        #dist2 = tf.sqrt(tf.reduce_mean(ndiff[:,:,:,:,0:16],axis=[2,3])) #NSC
        #dist3 = tf.sqrt(tf.reduce_mean(ndiff[:,:,:,:,16:31],axis=[2,3])) #NSC
        
        
        #dist2 = tf.reduce_sum(dist2/16,axis=2)# ^2.0
        #dist2 = tf.reduce_mean(dist2,axis=1) #N
        #dist2 = tf.reduce_mean(dist2)

        #dist3 = tf.reduce_sum(dist3/15,axis=2)# ^2.0
        #dist3 = tf.reduce_mean(dist3,axis=1)
        #dist3 = tf.reduce_mean(dist3)
        
        #dist2 = -10.0*dist2
        #dist = dist1 + 0.15*dist2
        #"""

        return dist2