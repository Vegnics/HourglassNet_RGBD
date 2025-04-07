import tensorflow as tf
import keras.losses
from hourglass_tensorflow.utils.tf import tf_batch_matrix_softargmax,tf_batch_multistage_matrix_softargmax_loss
from keras.utils import register_keras_serializable

@register_keras_serializable(package="mylosses")
class MAE_custom(keras.losses.Loss):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.AUTO, name="MAEcustom",WL2_j1: float = 1.0,WL2_j2: float = 0.3, WCoords: float = 0.001, *args, **kwargs
    ):
        super().__init__(reduction="sum", name=name)
        self.stages = 3
        self.n1joints = 14
        self.n2joints = 12
        self.use2joints = True
        self.wl2_j1 = tf.constant(WL2_j1,dtype=tf.float32)
        self.wl2_j2 = tf.constant(WL2_j2,dtype=tf.float32)
        self.wcoords = tf.constant(WCoords,dtype=tf.float32)
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
        #"""
        #gt_coords = tf_batch_multistage_matrix_softargmax_loss(y_true[:,:,:,:,0:self.n1joints])#NSC
        #pred_coords = tf_batch_multistage_matrix_softargmax_loss(y_pred[:,:,:,:,0:self.n1joints])
        #loss_coords = tf.math.square(gt_coords-pred_coords)#NSC2
        #loss_coords = tf.reduce_sum(loss_coords,axis=-1) #NSC
        #loss_coords = tf.reduce_mean(loss_coords,axis=-1) #NS
        #loss_coords = tf.math.sqrt(tf.reduce_mean(loss_coords,axis=1))
        #loss_coords = tf.reduce_mean(loss_coords)
        mask_joints_1 = tf.where(tf.reduce_max(y_true[:,:,:,:,0:self.n1joints],axis=[2,3])>0.0001,1.0,0.0) #NSC
        joint_count_1 = tf.reduce_sum(mask_joints_1,axis=2)#NS
        mask_joints_2 = tf.where(tf.reduce_max(y_true[:,:,:,:,self.n1joints:self.n1joints+self.n2joints],axis=[2,3])>0.0001,1.0,0.0) #NSC
        joint_count_2 = tf.reduce_sum(mask_joints_2,axis=2)#NS
        ndiff = tf.math.square(y_true-y_pred)
        ndiff = tf.reduce_mean(ndiff,axis=[2,3]) #NSHW  0.00000001
        loss_1jnt = (tf.reduce_sum(ndiff[:,:,0:self.n1joints]*mask_joints_1,axis=2))/(tf.cast(joint_count_1,dtype=tf.float32)+0.001) #NS
        loss_1jnt = tf.reduce_mean(loss_1jnt,axis=1)
        loss_2jnt = (tf.reduce_sum(ndiff[:,:,self.n1joints:self.n1joints+self.n2joints]*mask_joints_2,axis=2))/(tf.cast(joint_count_2,dtype=tf.float32)+0.001)#NS
        loss_2jnt = tf.reduce_mean(loss_2jnt,axis=1)
        Loss_final = self.wl2_j1*tf.reduce_mean(loss_1jnt)+self.wcoords*0.0+self.wl2_j2*tf.cast(self.use2joints,dtype=tf.float32)*tf.reduce_mean(loss_2jnt)
        return Loss_final