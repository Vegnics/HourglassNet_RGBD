import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os,sys
sys.path.insert(1,os.getcwd())
from hourglass_tensorflow.handlers._transformation import tf_train_map_affine_augmentation,tf_train_map_squarify,tf_train_map_resize_data
from hourglass_tensorflow.utils.tf import tf_load_image,tf_rotate_tensor,tf_rotate_coords,tf_compute_bbox,tf_expand_bbox
from hourglass_tensorflow.utils.tf import tf_compute_bbox,tf_expand_bbox,tf_expand_bbox_squared


LM_NAMES = [ "rAnkle",
            "rKnee",
            "rHip",
            "lHip",
            "lKnee",
            "lAnkle",
            "pelvis",
            "thorax",
            "upperNeck",
            "topHead",
            "rWrist",
            "rElbow",
            "rShoulder",
            "lShoulder",
            "lElbow",
            "lWrist"]

LM_POS = [ 
(483, 572),
(524, 443),
(519, 299),
(625, 298),
(661, 444),
(658, 568),
(576, 307),
(572, 180),
(576, 144),
(573, 77),
(414, 248),
(490, 242),
(518, 165),
(630, 155),
(670, 206),
(714, 276)
]

LM_POS =[
(480, 456),
(460, 446),
(397, 390),
(484, 347),
(536, 456),
(545, 427),
(457, 379),
(417, 294),
(405, 261),
(361, 146),
(278, 133),
(275, 208),
(330, 238),
(470, 213),
(515, 262),
(539, 331)
]

#"""
img_tf = tf_load_image("data/test_swimmer.png")
#"""
#H,W,_ = img_tf.shape()
coordinates = tf.convert_to_tensor(LM_POS,dtype=tf.int32)
#rcoordinates = tf.convert_to_tensor(LM_POS,dtype=tf.float32)/256.0
visibility = tf.constant([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
img,coords,vis = tf_train_map_squarify(img_tf,coordinates,visibility,bbox_enabled=True,bbox_factor=1.14)
img,coords,vis = tf_train_map_resize_data(img,coords,vis,input_size=256)
imgs,coords,vis = tf_train_map_affine_augmentation(img,coords,vis,input_size=256)
for i in range(imgs.shape[0]):
    r_img = imgs[i].numpy()
    #bbox = tf_compute_bbox(256.0*coords[i])
    #bbox = tf_expand_bbox_squared(bbox,tf.constant([256,256,3],dtype=tf.int32),1.155)
    #tx,ty,bx,by = int(bbox[0,0]),int(bbox[0,1]),int(bbox[1,0]),int(bbox[1,1])
    #cv2.rectangle(r_img,(tx,ty),(bx,by),(0,0,255),2)
    r_coords = 256.0*coords[i].numpy()
    for pnt in r_coords:
        center = (int(pnt[0]),int(pnt[1]))
        cv2.circle(r_img,center,3,(255,0,0),-1)
    plt.imshow(r_img)
    plt.show() 
#"""
"""
for angle in [-45,-30,-15,0,15,30,45]:    
    A = tf_rotate_tensor(img_tf,tf.constant(angle))
    r_img = A.numpy()
    C = tf_rotate_coords(coordinates,tf.constant(angle),256)
    r_coords = C.numpy()
    for pnt in r_coords:
        cv2.circle(r_img,pnt,3,(255,0,0),-1) 
    plt.imshow(r_img)
    plt.show()
"""
