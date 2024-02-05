import tensorflow as tf
import os,sys
sys.path.insert(1,os.getcwd())
from hourglass_tensorflow.handlers._transformation import tf_train_map_affine_augmentation
from hourglass_tensorflow.utils.tf import tf_load_image,tf_rotate_tensor,tf_rotate_coords,tf_compute_bbox,tf_expand_bbox
from hourglass_tensorflow.utils.tf import tf_compute_bbox,tf_expand_bbox,tf_expand_bbox_squared
import cv2
from matplotlib import pyplot as plt

LM_POS = [(49,220),
(87,176),
(110,122),
(135,114),
(172,158),
(162,211),
(126,118),
(142,74),
(135,46),
(134,21),
(194,63),
(158,69),
(128,55),
(148,55),
(146,94),
(180,94)]

shape = tf.constant([512.0,512.0],dtype=tf.float64)
input_size = tf.constant(256.0,dtype=tf.float64)
coordinates = tf.cast(2.0*tf.convert_to_tensor(LM_POS,dtype=tf.float64),dtype=tf.float64)
# We compute the Height and Width reduction factors
h_factor = shape[0] / tf.cast(input_size, tf.dtypes.float64)
w_factor = shape[1] / tf.cast(input_size, tf.dtypes.float64)
print((w_factor, h_factor)/input_size)
# We can recompute relative Coordinates between 0-1 as float
#rcoordinates = (
#    tf.cast(coordinates, dtype=tf.dtypes.float64)/(w_factor, h_factor)/input_size
#)

rcoordinates = (
    tf.cast(coordinates, dtype=tf.dtypes.float64)/(shape[1], shape[0])
) 
#print(coordinates)
#print(rcoordinates)

img_tf = tf_load_image("data/test_tennis.png")
visibility = tf.constant([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
imgs,coords,_ = tf_train_map_affine_augmentation(img_tf,tf.cast(rcoordinates,dtype=tf.float32),visibility,256)#,angles,shifts,256)
for i in range(imgs.shape[0]):
    r_img = imgs[i].numpy()
    bbox = tf_compute_bbox(256.0*coords[i])
    bbox = tf_expand_bbox_squared(bbox,tf.constant([256,256,3],dtype=tf.int32),1.16)
    tx,ty,bx,by = int(bbox[0,0]),int(bbox[0,1]),int(bbox[1,0]),int(bbox[1,1])
    cv2.rectangle(r_img,(tx,ty),(bx,by),(0,0,255),2)
    r_coords = 256.0*coords[i].numpy()
    for pnt in r_coords:
        center = (int(pnt[0]),int(pnt[1]))
        cv2.circle(r_img,center,3,(255,0,0),-1)
    plt.imshow(r_img)
    plt.show() 
