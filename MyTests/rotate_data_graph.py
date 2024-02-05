import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os,sys
sys.path.insert(1,os.getcwd())
from hourglass_tensorflow.handlers._transformation import tf_train_map_affine_augmentation
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

LM_POS = [ (49,220),
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

"""
tofloat = lambda x:(float(x[0]),float(x[1]))
center = tofloat(LM_POS[5])
img = cv2.imread("data/test_tennis.png")
M = cv2.getRotationMatrix2D(center,30.0,1.0)
img2 = cv2.warpAffine(img,M,(256,256))
_LM_POS = np.array(LM_POS,dtype=np.float32)

R_LM_POS = np.floor(M[0:2,0:2]@_LM_POS.T + np.reshape(M[:,2],(-1,1))).astype(np.int32)
R_LM_POS = R_LM_POS.T

for pnt in R_LM_POS:
    cv2.circle(img2,pnt,4,(0,0,255),-1) 

plt.imshow(img2[:,:,::-1])
plt.show()
"""
#"""
_center = LM_POS[6]
cx = _center[0]
cy = _center[1]
img_tf = tf_load_image("data/test_tennis.png")
angles = [-15,0,15]
shifts = [-10,0,10]
coordinates = tf.convert_to_tensor(LM_POS,dtype=tf.float32)
rcoordinates = tf.convert_to_tensor(LM_POS,dtype=tf.float32)/256.0
visibility = tf.constant([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
imgs,coords,_ = tf_train_map_affine_augmentation(img_tf,rcoordinates,visibility,256)#,angles,shifts,256)
for i in range(imgs.shape[0]):
    r_img = imgs[i].numpy()
    bbox = tf_compute_bbox(256.0*coords[i])
    bbox = tf_expand_bbox_squared(bbox,tf.constant([256,256,3],dtype=tf.int32),1.155)
    tx,ty,bx,by = int(bbox[0,0]),int(bbox[0,1]),int(bbox[1,0]),int(bbox[1,1])
    cv2.rectangle(r_img,(tx,ty),(bx,by),(0,0,255),2)
    r_coords = 256.0*coords[i].numpy()
    for pnt in r_coords:
        center = (int(pnt[0]),int(pnt[1]))
        cv2.circle(r_img,center,3,(255,0,0),-1)
    plt.imshow(r_img)
    plt.show() 

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
