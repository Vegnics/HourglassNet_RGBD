import tensorflow as tf

# Sample tensor
Xs = tf.constant([-5,1.0, 2.0, 3.0], dtype=tf.float32)



# Cast the tensor within a GPU context
with tf.device('/GPU:0'):
    viszeros= tf.constant([0]*4,dtype=tf.float32)
    visinf= tf.constant([100]*4,dtype=tf.float32)
    visx = tf.where(Xs>-4,viszeros,visinf)
    minx = tf.reduce_min(Xs+visx)
    casted_tensor = tf.cast(Xs, dtype=tf.int32)

print(casted_tensor,minx)

# Enable logging device placement
tf.debugging.set_log_device_placement(True)

# Another cast operation to observe device placement logs
with tf.device('/GPU:0'):
    viszeros= tf.constant([0]*4,dtype=tf.float32)
    visinf= tf.constant([100]*4,dtype=tf.float32)
    visx = tf.where(Xs>-4,viszeros,visinf)
    minx = tf.reduce_min(Xs+visx)
    another_casted_tensor = tf.cast(Xs, dtype=tf.int32)

print(another_casted_tensor,minx)