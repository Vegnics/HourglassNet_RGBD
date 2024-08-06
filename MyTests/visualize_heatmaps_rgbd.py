import tensorflow as tf
import keras
import sys,os
import sys,os
from matplotlib import pyplot as plt
sys.path.insert(1,os.getcwd())
import numpy as np
import cv2 
import csv

from hourglass_tensorflow.metrics.correct_keypoints import *
from hourglass_tensorflow.losses.mae_custom import *
from hourglass_tensorflow.utils.tf import tf_load_image,tf_3Uint8_to_float32
from hourglass_tensorflow.handlers._transformation import tf_train_map_squarify


def read_landmark_data(csv_path: str):
    with open(csv_path,"r") as csv_file:
        reader = csv.reader(csv_file)
        annopoints = [
            [int(drow[0]),int(drow[1])]
            for drow in reader
            ]
    return np.array(annopoints)

def draw_pose(img,hm):
    _img = np.copy(img)
    kpnts = []
    for i in range(14):
        pnt = np.argmax(hm[:,:,i])
        x = int(4*(pnt%64))
        y = int(4*(pnt//64))
        val = hm[int(pnt//64),int(pnt%64),i]
        print(f"Landmark {LM_NAMES[i]}:  ({x},{y})")
        kpnts.append([x,y,val])
    kpnts = np.array(kpnts)
    _visible_kpts = np.array([i for i in range(14)])
    _visible_kpts = list(_visible_kpts[kpnts[:,2]>0.3])
    KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (150,80,50),
    (1, 2): (150,80,50),
    (2, 3): (80,160,60),
    (3, 4): (50,80,160),
    (4, 5): (50,80,160),
    (6, 7): (150,80,50),
    (7, 8): (150,80,50),
    (8,12): (150,80,50),
    (9,12): (50,80,160),
    (9, 10): (50,80,160),
    (10, 11): (50,80,160),
    (2, 12): (150,80,50),
    (3, 12): (50,80,160),
    (12, 13): (230,10,20)
    }
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if edge_pair[0] in _visible_kpts and edge_pair[1] in _visible_kpts:
            x0=int(kpnts[edge_pair[0],0])
            y0=int(kpnts[edge_pair[0],1])
            x1=int(kpnts[edge_pair[1],0])
            y1=int(kpnts[edge_pair[1],1])
            cv2.line(_img,(x0,y0),(x1,y1),color,2)
    for pnt in kpnts:
        if pnt[2]>0.3:
            cv2.circle(_img,(int(pnt[0]),int(pnt[1])),4,(0,0,255),-1)
    return _img

def getbbox(landmarks):
    xs = landmarks[:,0]
    ys = landmarks[:,1]
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)
    return [[x_min,x_max],[y_min,y_max]]

def preprocess_img(img,depth,landmarks):
    bbox = getbbox(landmarks)
    pass



Model = tf.keras.models.load_model("data/model_t/myModel_SLP",
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

subject_id = 10
cover = "cover2"
img_num = 10
#img_bgr = cv2.imread("/home/quinoa/football_player.png")#cv2.imread("data/test_tennis.png")"/home/quinoa/tennis.png"
imgrgb = tf_load_image("/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD/{:05d}/RGB/{}/image_{:06d}.jpg".format(subject_id,cover,img_num))#tf_load_image("data/test_tennis.png")
imagedepth = tf_load_image("/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD/{:05d}/Depth/{}/depth_{:06d}.png".format(subject_id,cover,img_num))
depthmap = tf.expand_dims(tf.clip_by_value((tf_3Uint8_to_float32(imagedepth)-500.0)*(255.0/2500.0),0.0,255.0),axis=2)
RGBD_image = tf.concat([tf.cast(imgrgb,dtype=tf.float32),depthmap],axis=2)
landmarks = tf.convert_to_tensor(read_landmark_data("/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD/{:05d}/LMData/lm_{:06d}.csv".format(subject_id,img_num)))
visibilities = tf.convert_to_tensor([1]*14)
squared_rgbd = tf_train_map_squarify(RGBD_image,tf.cast(landmarks,tf.float32),visibilities,True,1.18)
tensor = tf.cast(tf.expand_dims(squared_rgbd[0],axis=0),dtype=tf.dtypes.float32)
#print()
#"""
img_bgr = np.uint8(tensor[0,:,:,0:3].numpy())
hms = Model.predict(tensor)
preds = hms[0,-1,:,:,:]
#hm = preds[:,:,:]
img_bgr = draw_pose(img_bgr,preds)
plt.imshow(img_bgr)#[:,:,::-1])
plt.show()
#"""
