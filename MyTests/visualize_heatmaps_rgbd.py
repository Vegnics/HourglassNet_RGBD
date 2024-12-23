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
from hourglass_tensorflow.handlers._transformation import tf_train_map_squarify,tf_test_map_affine_woaugment_RGBD
from hourglass_tensorflow.metrics.distance import OverallMeanDistance

def read_landmark_data(csv_path: str):
    with open(csv_path,"r") as csv_file:
        reader = csv.reader(csv_file)
        annopoints = [
            [int(drow[0]),int(drow[1])]
            for drow in reader
            ]
    return np.array(annopoints)



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


def get_secondmax(hm,x,y):
    n = 1.0*hm[y-1:y+2,x-1:x+2]
    m = tf.convert_to_tensor([[1,1,1],
                              [1,0,1],
                              [1,1,1]],dtype=tf.float32)
    _n = n*m
    pnt = np.argmax(_n)
    dx = int(pnt%3)-1
    dy = int(pnt//3)-1
    if hm[y+dy,x+dx]>0.38:
        return (dx,dy)
    else:
        return (0,0)
    

def draw_poseGT(img,landmarks):
    _img = np.copy(img)
    kpnts = []
    for i in range(14):
        pnt = landmarks[i]
        x = int((pnt[0]))
        y = int((pnt[1]))
        val = 1.0
        kpnts.append([x,y,val])
    kpnts = np.array(kpnts)
    _visible_kpts = np.array([i for i in range(14)])
    _visible_kpts = list(_visible_kpts[kpnts[:,2]>0.45])
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
            cv2.line(_img,(x0,y0),(x1,y1),color,5)
    for pnt in kpnts:
        if pnt[2]>0.45:
            cv2.circle(_img,(int(pnt[0]),int(pnt[1])),9,(0,0,255),-1)
    return _img

    
def draw_pose(img,hm,obbox,pad):
    _img = np.copy(img)
    #bbox[0, 0] : bbox[1, 0]
    Hb = obbox[1,1] - obbox[0,1]
    Wb = obbox[1,0] - obbox[0,0]
    N = max(Wb,Hb)
    padx = pad[0]*64/N
    pady = pad[1]*64/N
    print(Wb,Hb,N,padx,pady)
    kpnts = []
    for i in range(14):
        pnt = np.argmax(hm[:,:,i])
        x = int((pnt%64))
        y = int((pnt//64))
        dx,dy = get_secondmax(hm[:,:,i],x,y)
        #x = 4*int((pnt%64) + 0.5*dx)
        #y = 4*int((pnt//64) + 0.5*dy)
        x = int((N/64.0)*((pnt%64) + 0.25*dx - padx)+ obbox[0,0])
        y = int((N/64.0)*((pnt//64) + 0.25*dy - pady)+ obbox[0,1])
        val = hm[int(pnt//64),int(pnt%64),i]
        print(f"Landmark {LM_NAMES[i]}:  ({x},{y},{val})")
        kpnts.append([x,y,val])
    kpnts = np.array(kpnts)
    _visible_kpts = np.array([i for i in range(14)])
    _visible_kpts = list(_visible_kpts[kpnts[:,2]>0.45])
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
            cv2.line(_img,(x0,y0),(x1,y1),color,5)
    for pnt in kpnts:
        if pnt[2]>0.45:
            cv2.circle(_img,(int(pnt[0]),int(pnt[1])),9,(0,0,255),-1)
    return _img
    
Model = tf.keras.models.load_model("data/model_t/myModel_SLP_fAB2",
                           custom_objects= {"OverallMeanDistance":OverallMeanDistance
                                            ,"PercentageOfCorrectKeypoints":PercentageOfCorrectKeypoints,
                                            "MAE_custom":MAE_custom})



            

LM_NAMES = ["00_rAnkle",
        "01_rKnee",
        "02_rHip",
        "03_lHip",
        "04_lKnee",
        "05_lAnkle",
        "06_rWrist",
        "07_rElbow",
        "08_rShoulder",
        "09_lShoulder",
        "10_lElbow",
        "11_lWrist",
        "12_thorax",
        "13_topHead"]

hm_scale = tf.constant(2.1269474)

subject_id = 5  
cover = "uncover"
img_num = 27
#img_bgr = cv2.imread("/home/quinoa/football_player.png")#cv2.imread("data/test_tennis.png")"/home/quinoa/tennis.png"
imgrgb = tf_load_image("/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD/{:05d}/RGB/{}/image_{:06d}.jpg".format(subject_id,cover,img_num))#tf_load_image("data/test_tennis.png")
imagedepth = tf_load_image("/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD/{:05d}/Depth/{}/depth_{:06d}.png".format(subject_id,cover,img_num))
depthmap = tf.expand_dims(tf_3Uint8_to_float32(imagedepth),axis=2)
meandepth = tf.reduce_mean(depthmap,axis=[0,1,2])
stddevdepth =  tf.sqrt(tf.reduce_mean(tf.square(depthmap-meandepth),axis=[0,1,2]))
depthmap = 1.5*((depthmap-meandepth)/stddevdepth+2.1)

RGBD_image = depthmap #tf.concat([tf.cast(imgrgb,dtype=tf.float32),depthmap],axis=2)
landmarks = tf.convert_to_tensor(read_landmark_data("/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD/{:05d}/LMData/lm_{:06d}.csv".format(subject_id,img_num)))
visibilities = tf.convert_to_tensor([1]*14)
squared_rgbd = tf_test_map_affine_woaugment_RGBD(RGBD_image,RGBD_image.shape,landmarks,visibilities,njoints=14)
#squared_rgbd = tf_train_map_squarify(RGBD_image,tf.cast(landmarks,tf.float32),visibilities,True,1.18)
#obbox = tf.cast(squared_rgbd[3],tf.float32)
obbox = tf.cast(squared_rgbd[2][0],tf.float32)
print(landmarks.shape,squared_rgbd[1][0].shape)

_obbox = obbox[0:2,0:2]

padding = obbox[2,0:2]
print(obbox)

tensor = squared_rgbd[0] #tf.cast(tf.expand_dims(squared_rgbd[0],axis=0),dtype=tf.dtypes.float32)
#print()
#"""
#img_bgr = np.uint8(tensor[0,:,:,0:3].numpy())
hms = Model.predict(tensor)
preds = hms[0,2,:,:,:]
normpreds = tf.linalg.norm(preds,axis=[0,1])
normpreds = tf.expand_dims(normpreds,axis=0)
normpreds = tf.expand_dims(normpreds,axis=0)
print(normpreds.shape)
preds = preds*hm_scale/normpreds

#hm = preds[:,:,:]
for i in range(14):
    hm = preds[:,:,i]
    pnt = np.argmax(hm)
    x = int(pnt%64)
    y = int(pnt//64)
    dx,dy = (0,0) #get_secondmax(hm,x,y)
    x = 4*int((pnt%64) + 0.5*dx)
    y = 4*int((pnt//64) + 0.5*dy)
    #cv2.circle(img_bgr,(x,y),5,(0,0,255),-1)
    print(f"Landmark {LM_NAMES[i]}:  ({x},{y})")
    #plt.imshow(hm,cmap="jet")
    #plt.savefig(f"/home/quinoa/{LM_NAMES[i]}_stage0.png", bbox_inches='tight')
    #plt.show()
img_bgr = draw_pose(imgrgb,preds,_obbox,padding)
img_gt = draw_poseGT(imgrgb,squared_rgbd[1][0])
plt.imshow(img_bgr)#[:,:,::-1])
plt.figure()
plt.imshow(img_gt)
#plt.savefig(f"/home/quinoa/sub_{subject_id}-num_{img_num}-{cover}.png", bbox_inches='tight')
plt.show()
#"""
