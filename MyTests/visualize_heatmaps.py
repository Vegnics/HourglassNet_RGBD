import tensorflow as tf
import keras
import sys,os
import sys,os
from matplotlib import pyplot as plt
sys.path.insert(1,os.getcwd())
import numpy as np
import cv2 

from hourglass_tensorflow.metrics.correct_keypoints import *
from hourglass_tensorflow.losses.mae_custom import *
from hourglass_tensorflow.utils.tf import tf_load_image


def draw_pose(img,hm):
    _img = np.copy(img)
    kpnts = []
    for i in range(16):
        pnt = np.argmax(hm[:,:,i])
        x = int(4*(pnt%64))
        y = int(4*(pnt//64))
        val = hm[int(pnt//64),int(pnt%64),i]
        print(f"Landmark {LM_NAMES[i]}:  ({x},{y})")
        kpnts.append([x,y,val])
    kpnts = np.array(kpnts)
    _visible_kpts = np.array([i for i in range(16)])
    _visible_kpts = list(_visible_kpts[kpnts[:,2]>0.2])
    KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (150,80,80), #rankle rknee
    (1, 2): (150,80,80), #rknee rhip 
    (2, 3): (100,160,90), #rhip lhip 
    (2, 12): (150,80,80), #rhip rshoulder
    (3, 4): (50,80,160), #lhip lknee
    (3, 13): (50,80,160), #lhip lshouder
    (4, 5): (50,80,160), #lknee lankle
    (6, 7): (150,80,80), #pelvis thorax
    (7, 8): (80,190,70), #thorax upperneck
    (7,12): (150,80,80), # thorax rshoulder
    (7,13): (50,80,160), # thorax lshoulder
    (8, 9): (90,150,70), # upperneck tophead
    (10,11): (150,80,80), #rwrist relbow
    (11,12): (150,80,80), #relbow rshoulder
    (13, 14): (50,80,160),#lshoulder lelbow
    (14, 15): (50,80,160) #lelbow lwrist
    }
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if edge_pair[0] in _visible_kpts and edge_pair[1] in _visible_kpts:
            x0=int(kpnts[edge_pair[0],0])
            y0=int(kpnts[edge_pair[0],1])
            x1=int(kpnts[edge_pair[1],0])
            y1=int(kpnts[edge_pair[1],1])
            cv2.line(_img,(x0,y0),(x1,y1),color,2)
    for pnt in kpnts:
        if pnt[2]>0.2:
            cv2.circle(_img,(int(pnt[0]),int(pnt[1])),4,(0,0,255),-1)
    return _img


Model = tf.keras.models.load_model("data/model_t/myModel_d",
                           custom_objects= {"RatioCorrectKeypoints":RatioCorrectKeypoints
                                            ,"PercentageOfCorrectKeypoints":PercentageOfCorrectKeypoints,
                                            "MAE_custom":MAE_custom})

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

img_path = "/home/quinoa/skater5.png" #"data/test_tennis.png" #"/home/quinoa/tenniscrop1.png" #

img_bgr = cv2.imread(img_path)#cv2.imread("/home/quinoa/football_player.png")#c #"/home/quinoa/tennis.png"
img = tf_load_image(img_path) #tf_load_image("/home/quinoa/football_player.png")#
tensor = tf.cast(tf.expand_dims(img,axis=0),dtype=tf.dtypes.float32)
#print(tensor)
hms = Model.predict(tensor)
preds = hms[0,-1,:,:,:]

img_bgr = np.copy(img_bgr[:,:,::-1])
img_bgr = draw_pose(img,preds)
"""
for i in range(16):
    hm = preds[:,:,i]
    pnt = np.argmax(hm)
    x = int(4*(pnt%64))
    y = int(4*(pnt//64))
    cv2.circle(img_bgr,(x,y),5,(0,0,255),-1)
    print(f"Landmark {LM_NAMES[i]}:  ({x},{y})")
    plt.imshow(hm,cmap="jet")
    plt.show()
"""
plt.imshow(img_bgr)#[:,:,::-1])
plt.show()

