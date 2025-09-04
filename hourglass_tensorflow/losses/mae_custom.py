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
        #tf.print(self.stages,self.n1joints,self.n2joints,self.use2joints)
        #tf.print(y_true)
        #tf.print(y_pred)
        #_y_pred = y_pred[0]
        S = self.stages
        #C = self.njoints
        #"""

        #Attention auxiliary loss (cosine similarity)
        gt_att = y_true[:,:,:,:,self.n1joints+self.n2joints+1:] #NSHWC
        pred_att = tf.nn.relu(y_pred[:,:,:,:,self.n1joints+self.n2joints+1:]) #NSHWC
        cossim = tf.reduce_sum(gt_att*pred_att,axis=[2,3])/(tf.linalg.norm(gt_att,axis=[2,3])*tf.linalg.norm(pred_att,axis=[2,3])+1e-6) #NSHW
        #attdiff = tf.math.square(gt_att-pred_att)#*(1.0+0.4*y_true) #NSHWC
        attdiff = tf.maximum(1.0 - cossim,0)
        attdiff = tf.reduce_mean(attdiff,axis=-1)
        #NSH
        #attdiff = tf.reduce_mean(1/tf.maximum(attdiff,1e-4),axis=-1)
        Aux_loss = tf.reduce_mean(attdiff)

        #Coordinate regression loss
        gt_coords = tf_batch_multistage_matrix_softargmax_loss(y_true[:,:,:,:,0:self.n1joints])#NSC
        pred_coords = tf_batch_multistage_matrix_softargmax_loss(y_pred[:,:,:,:,0:self.n1joints])
        loss_coords = tf.math.square(gt_coords-pred_coords)#NSC2
        loss_coords = tf.reduce_sum(loss_coords,axis=-1) #NSC
        loss_coords = tf.reduce_mean(loss_coords,axis=-1) #NS
        loss_coords = tf.reduce_mean(loss_coords,axis=1)#N
        loss_coords = tf.reduce_mean(loss_coords)
        #loss_coords = tf.reduce_mean(tf.minimum(loss_coords,7.0))
        #tf.debugging.check_numerics(y_pred,"y_pred has invalid numeric values")
        #tf.debugging.check_numerics(y_true,"y_true has invalid numeric values")
        mask_joints_1 = tf.where(tf.reduce_max(y_true[:,-1,:,:,0:self.n1joints],axis=[1,2])>0.0001,1.0,0.0) #NC
        joint_count_1 = tf.reduce_sum(mask_joints_1,axis=1)#N
        mask_joints_2 = tf.where(tf.reduce_max(y_true[:,-1,:,:,self.n1joints:self.n1joints+self.n2joints],axis=[1,2])>0.0001,1.0,0.0) #NC
        joint_count_2 = tf.reduce_sum(mask_joints_2,axis=1)#N

        heatmap_weights = tf.where(y_true>0.001,1.25,1.0) #NSHWC

        # Attention loss
        #attention_maps = self.model.get_layer("Hourglass2").capture #NSHWC
        #tf.print(tf.shape(attention_maps))
        # GT and pred sums
        mag_true =  tf.reduce_sum(y_true[:,:,:,:,0:self.n1joints],axis=[2,3])/64.0 #N,S,C
        mag_pred = tf.reduce_sum(y_pred[:,:,:,:,0:self.n1joints],axis=[2,3])/64.0 #N,S,C
        
        # Heatmap regression losses
        ndiff = tf.math.square(y_true-y_pred)*heatmap_weights#*(1.0+0.4*y_true) #NSHWC
        ndiff = tf.reduce_mean(ndiff,axis=[1,2,3]) #NC  0.00000001
        #tf.debugging.check_numerics(ndiff,"ndiff has invalid numeric values")
        
        # 1J Heatmap regression loss
        loss_1jnt = (tf.reduce_sum(ndiff[:,0:self.n1joints]*mask_joints_1,axis=1))/(tf.cast(joint_count_1,dtype=tf.float32)+0.001) #NS
        #tf.debugging.check_numerics(loss_1jnt,"loss_1jnt has invalid numeric values")
        #loss_1jnt = tf.reduce_mean(loss_1jnt,axis=1)
        
        # 2J Heatmap regression loss
        loss_2jnt = (tf.reduce_sum(ndiff[:,self.n1joints:self.n1joints+self.n2joints]*mask_joints_2,axis=1))/(tf.cast(joint_count_2,dtype=tf.float32)+0.001)#NS
        
        # Cumulative heatmap regression loss
        cum_diff = tf.square(mag_true-mag_pred) #N,S,C
        cum_diff = tf.reduce_mean(cum_diff,axis=1) #N,C
        cum_loss = (tf.reduce_sum(cum_diff*tf.cast(mask_joints_1,dtype=tf.float32),axis=-1))/(tf.cast(joint_count_1,dtype=tf.float32)+0.001)
        cum_loss = tf.reduce_mean(cum_loss)
        
        #tf.debugging.check_numerics(loss_2jnt,"loss_2jnt has invalid numeric values")
        #loss_2jnt = tf.reduce_mean(loss_2jnt,axis=1)
        Loss_final = self.wl2_j1*tf.reduce_mean(loss_1jnt)#+self.wcoords*loss_coords+self.wl2_j2*tf.cast(self.use2joints,dtype=tf.float32)*tf.reduce_mean(loss_2jnt)
        return Loss_final + 0.00003*Aux_loss #+ 1e-3*cum_loss