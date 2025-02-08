import tensorflow as tf
import keras.metrics

from hourglass_tensorflow.utils.tf import tf_dynamic_matrix_argmax, tf_batch_matrix_softargmax

def softargmax_2d(heatmap):
    """
    Applies Softargmax to a 2D heatmap to extract (x, y) coordinates.
    """
    #_ , h, w, c = tf.shape(heatmap)
    h = 64
    w = 64
    c = 14
    
    # Flatten spatial dimensions
    heatmap_flat = tf.reshape(heatmap, (-1, h * w, c))  

    # Apply softmax to normalize heatmaps
    heatmap_flat = tf.nn.softmax(heatmap_flat, axis=1) #NxHWxC 

    # Create coordinate grids
    x_grid = tf.range(w, dtype=tf.float32)
    y_grid = tf.range(h, dtype=tf.float32)
    x_grid, y_grid = tf.meshgrid(x_grid, y_grid)

    # Flatten coordinate grids
    x_grid = tf.reshape(x_grid, shape=(1,w*h,1))  
    y_grid = tf.reshape(y_grid, shape=(1,w*h,1))  

    # Compute expected (x, y) coordinates using softmax weights
    x = tf.reduce_sum(x_grid * heatmap_flat, axis=1) #NxC
    y = tf.reduce_sum(y_grid * heatmap_flat, axis=1) #NxC  

    # Stack coordinates (NxCx2)
    coords = tf.stack([y, x], axis=-1)  
    return coords

class SoftargmaxMeanDist(keras.metrics.Metric):
    def __init__(self, name="softargmax_medist",  intermediate_supervision: bool = True,num_1joints: int = 16, **kwargs):
        super(SoftargmaxMeanDist, self).__init__(name=name, **kwargs)
        self.intermediate_supervision = intermediate_supervision
        self.cum_mean_distance = self.add_weight(name="cum_mean_distance", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.num_1joints = num_1joints
    
    def argmax_tensor(self, tensor):
        return tf_dynamic_matrix_argmax(
            tensor[:,:,:,:,0:self.num_1joints],
            intermediate_supervision=self.intermediate_supervision,
            keepdims=True,
        )
    
    def check_visibility(self,tensor):
        _tensor = 1.0*tensor[:,-1,:,:,:]
        sum = tf.reduce_max(_tensor,axis=[1,2])# NC
        vis0 = tf.zeros_like(sum)
        vis1 = tf.ones_like(sum)
        vis = tf.where(sum<0.8,vis0,vis1)#NC
        return vis

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        y_true: Ground truth keypoint coordinates (bs, num_joints, 2)
        y_pred: Predicted heatmaps (bs, h, w, num_joints)
        """
        vis = self.check_visibility(y_true[:,:,:,:,0:self.num_1joints])
        Njoints = tf.reduce_sum(vis)
        # Convert heatmaps to coordinates
        ground_truth_joints = tf.cast(self.argmax_tensor(y_true),dtype=tf.float32)
        #pred_coords = tf.cast(self.argmax_tensor(y_true),dtype=tf.float32)
        pred_coords = tf_batch_matrix_softargmax(y_pred[:,-1,:,:,0:self.num_1joints])
        diff = tf.cast(
            ground_truth_joints - pred_coords, dtype=tf.dtypes.float32
        ) #NxCx2
        # Compute Euclidean distance
        dist = tf.norm(diff, axis=-1)  # NxC
        cum_distance = tf.reduce_sum(dist*vis, axis=1) 
        mean_distance = tf.reduce_sum(cum_distance)/tf.cast(Njoints,dtype=tf.float32)
        self.cum_mean_distance.assign_add(mean_distance)
        self.count.assign_add(1.0)

    def result(self):
        return self.cum_mean_distance / self.count  # Mean Euclidean Distance

    def reset_state(self):
        self.cum_mean_distance.assign(0)
        self.count.assign(0)

class OverallMeanDistance(keras.metrics.Metric):
    def __init__(
        self, name=None, dtype=None, intermediate_supervision: bool = True, num_1joints: int = 16, **kwargs
    ):
        super().__init__(name, dtype, **kwargs)
        self.num_samples = self.add_weight(name="number_of_samples", initializer="zeros")
        self.distance = self.add_weight(name="distance", initializer="zeros")
        self.batch_mode = False
        self.intermediate_supervision = intermediate_supervision
        self.num_1joints = num_1joints

    def check_visibility(self,tensor):
        _tensor = 1.0*tensor[:,-1,:,:,:]
        sum = tf.reduce_max(_tensor,axis=[1,2])# NC
        vis0 = tf.zeros_like(sum)
        vis1 = tf.ones_like(sum)
        vis = tf.where(sum<0.8,vis0,vis1)#NC
        return vis

    def argmax_tensor(self, tensor):
        return tf_dynamic_matrix_argmax(
            tensor[:,:,:,:,0:self.num_1joints],
            intermediate_supervision=self.intermediate_supervision,
            keepdims=True,
        )

    def _internal_update(self, y_true, y_pred):
        vis = self.check_visibility(y_true[:,:,:,:,0:self.num_1joints])
        N = tf.ones_like(vis,dtype=tf.float32)
        N = tf.reduce_sum(N)/14.0

        #Njoints = tf.reduce_sum(vis,axis=1)
        Njoints = tf.reduce_sum(vis)
        ground_truth_joints = self.argmax_tensor(y_true)
        predicted_joints = self.argmax_tensor(y_pred)
        distance = tf.cast(
            ground_truth_joints - predicted_joints, dtype=tf.dtypes.float32
        )
         # We compute the norm of the reference limb from the ground truth
        reference_limb_error = tf.cast(
            ground_truth_joints[:, 12, :]
            - ground_truth_joints[:, 13, :],
            dtype=tf.float32,
        )# Nx2
        reference_distance = tf.norm(reference_limb_error, ord=2, axis=-1) #N
        reference_distance = tf.expand_dims(reference_distance,axis=1) #Nx1

        err_distance = tf.norm(distance, ord=2, axis=-1)#NC
        err_distance = err_distance
        #perc_distance = 100.0*tf.math.divide_no_nan(err_distance,reference_distance)
        perc_distance = tf.reduce_sum(err_distance*vis, axis=1) #/tf.cast(Njoints,dtype=tf.float32)
        #cum_mean_perc_distance = tf.reduce_sum(perc_distance/Njoints) 
        cum_mean_perc_distance = tf.reduce_sum(perc_distance)/tf.cast(Njoints,dtype=tf.float32)
        self.distance.assign_add(cum_mean_perc_distance)
        #self.num_samples.assign_add(N)
        self.num_samples.assign_add(1.0)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        return self._internal_update(y_true, y_pred)

    def result(self):
        return tf.math.divide_no_nan(self.distance, self.num_samples)

    def reset_state(self) -> None:
        self.num_samples.assign(0.0)
        self.distance.assign(0.0)
