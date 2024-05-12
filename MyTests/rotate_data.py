import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os,sys
sys.path.insert(1,os.getcwd())
from hourglass_tensorflow.handlers._transformation import tf_train_map_affine_augmentation
from hourglass_tensorflow.utils.tf import tf_load_image,tf_rotate_tensor
from keras.preprocessing.image import apply_affine_transform 


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

LM_POS =[ #swimmer
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
#cx = 128
#cy = 128
img_tf = tf_load_image("data/test_swimmer.png")
angles = [-15,0,15]
shifts = [-10,0,10]
coordinates = tf.convert_to_tensor(LM_POS)
visibility = tf.constant([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
print(img_tf.shape)
A = tf_train_map_affine_augmentation(img_tf,img_tf.shape,coordinates,visibility)#,angles,shifts,256)
print(tf.shape(A[0]))

for img,coord in zip(A[0],A[1]):
    _img = img.numpy()
    _coord = coord.numpy()
    for pnt in _coord:
        print(pnt)
        cv2.circle(_img,(int(pnt[0]),int(pnt[1])),5,(255,0,0),-1) 
    plt.imshow(_img)
    plt.show()
"""
M = cv2.getRotationMatrix2D((128,128),float(angle),1.0)
tx = -20
ty = 0
t = np.array([[-tx,-ty]]).T
_LM_POS = np.array(LM_POS,dtype=np.float32)
R_LM_POS = np.floor(M[0:2,0:2]@_LM_POS.T + np.reshape(M[:,2],(-1,1)) + t ).astype(np.int32)
R_LM_POS = R_LM_POS.T

img_t = tf.transpose(img_tf,perm=[2,0,1])
img_rotated = apply_affine_transform(img_t,angle,tx,ty,0,1,1,1,2,0,"nearest")
img_rotated = tf.transpose(img_rotated,perm=[1,2,0])
_img_rotated = img_rotated.numpy()

for pnt in R_LM_POS:
    cv2.circle(_img_rotated,pnt,3,(255,0,0),-1) 

plt.imshow(_img_rotated)
plt.show()
"""