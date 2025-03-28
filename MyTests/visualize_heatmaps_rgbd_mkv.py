import tensorflow as tf
import keras
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
from hourglass_tensorflow.utils.loaders.model_loader import load_wrapped_model

def read_landmark_data(csv_path: str):
    with open(csv_path,"r") as csv_file:
        reader = csv.reader(csv_file)
        annopoints = [
            [int(drow[0]),int(drow[1])]
            for drow in reader
            ]
    return np.array(annopoints)

def read_landmarks(fname):
    lm_array = np.zeros((18,2),dtype=np.int32)
    vis_array = np.zeros((18,),dtype=np.float32)
    with open(fname,"r") as file:
        reader = csv.reader(file)
        for r,row in enumerate(reader):
            if float(row[2])==0:
                lm_array[r,:] = np.array([-100000000,-100000000])
            else:
                lm_array[r,:] = np.array([int(row[0]),int(row[1])])
            vis_array[r] = float(row[2])
    return lm_array,vis_array



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
    for i in range(18):
        pnt = landmarks[i]
        x = int((pnt[0]))
        y = int((pnt[1]))
        val = 1.0
        kpnts.append([x,y,val])
    kpnts = np.array(kpnts)
    _visible_kpts = np.array([i for i in range(18)])
    _visible_kpts = list(_visible_kpts[kpnts[:,2]>0.45])
    KEYPOINT_EDGE_INDS_TO_COLOR = {
    # Upper body / torso
    (0, 1): (150, 100, 0),   # Nose to Neck
    (1, 2): (200, 160, 50),   # Neck to Right Shoulder
    (1, 5): (50, 160, 200),   # Neck to Left Shoulder

    # Right arm
    (2, 3): (200, 160, 50),   # Right Shoulder to Right Elbow
    (3, 4): (200, 160, 50),   # Right Elbow to Right Wrist

    # Left arm
    (5, 6): (50, 160, 200),   # Left Shoulder to Left Elbow
    (6, 7): (50, 160, 200),   # Left Elbow to Left Wrist

    # Torso to legs (using Neck as central hub)
    (1, 8): (200, 160, 50),   # Neck to Right Hip
    (1, 11): (50, 160, 200),  # Neck to Left Hip

    # Right leg
    (8, 9): (200, 160, 50),   # Right Hip to Right Knee
    (9, 10): (200, 160, 50),  # Right Knee to Right Ankle

    # Left leg
    (11, 12): (50, 160, 200), # Left Hip to Left Knee
    (12, 13): (50, 160, 200), # Left Knee to Left Ankle (accent color)

    # Additional pelvis connection
    (8, 11): (0, 150, 150),  # Right Hip to Left Hip

    # Face connections
    (0, 14): (200, 160, 50),  # Nose to Right Eye
    (0, 15): (50, 160, 200),  # Nose to Left Eye
    (14, 16): (200, 160, 50), # Right Eye to Right Ear
    (15, 17): (50, 160, 200)  # Left Eye to Left Ear
}
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if edge_pair[0] in _visible_kpts and edge_pair[1] in _visible_kpts:
            x0=int(kpnts[edge_pair[0],0])
            y0=int(kpnts[edge_pair[0],1])
            x1=int(kpnts[edge_pair[1],0])
            y1=int(kpnts[edge_pair[1],1])
            cv2.line(_img,(x0,y0),(x1,y1),color,3)
    for pnt in kpnts:
        if pnt[2]>0.45:
            cv2.circle(_img,(int(pnt[0]),int(pnt[1])),6,(0,0,255),-1)
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
    for i in range(18):
        pnt = np.argmax(hm[:,:,i])
        x = int((pnt%64))
        y = int((pnt//64))
        dx,dy = (0,0)#get_secondmax(hm[:,:,i],x,y)
        #x = 4*int((pnt%64) + 0.5*dx)
        #y = 4*int((pnt//64) + 0.5*dy)
        x = int((N/64.0)*((pnt%64) + 0.0*dx - padx)+ obbox[0,0])
        y = int((N/64.0)*((pnt//64) + 0.0*dy - pady)+ obbox[0,1])
        val = hm[int(pnt//64),int(pnt%64),i]
        #print(f"Landmark {LM_NAMES[i]}:  ({x},{y},{val})")
        kpnts.append([x,y,val])
    kpnts = np.array(kpnts)
    _visible_kpts = np.array([i for i in range(18)])
    _visible_kpts = list(_visible_kpts[kpnts[:,2]>0.1])
    KEYPOINT_EDGE_INDS_TO_COLOR = {
    # Upper body / torso
    (0, 1): (150, 100, 0),   # Nose to Neck
    (1, 2): (200, 160, 50),   # Neck to Right Shoulder
    (1, 5): (50, 160, 200),   # Neck to Left Shoulder

    # Right arm
    (2, 3): (200, 160, 50),   # Right Shoulder to Right Elbow
    (3, 4): (200, 160, 50),   # Right Elbow to Right Wrist

    # Left arm
    (5, 6): (50, 160, 200),   # Left Shoulder to Left Elbow
    (6, 7): (50, 160, 200),   # Left Elbow to Left Wrist

    # Torso to legs (using Neck as central hub)
    (1, 8): (200, 160, 50),   # Neck to Right Hip
    (1, 11): (50, 160, 200),  # Neck to Left Hip

    # Right leg
    (8, 9): (200, 160, 50),   # Right Hip to Right Knee
    (9, 10): (200, 160, 50),  # Right Knee to Right Ankle

    # Left leg
    (11, 12): (50, 160, 200), # Left Hip to Left Knee
    (12, 13): (50, 160, 200), # Left Knee to Left Ankle (accent color)

    # Additional pelvis connection
    (8, 11): (0, 150, 150),  # Right Hip to Left Hip

    # Face connections
    (0, 14): (200, 160, 50),  # Nose to Right Eye
    (0, 15): (50, 160, 200),  # Nose to Left Eye
    (14, 16): (200, 160, 50), # Right Eye to Right Ear
    (15, 17): (50, 160, 200)  # Left Eye to Left Ear
}
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if edge_pair[0] in _visible_kpts and edge_pair[1] in _visible_kpts:
            x0=int(kpnts[edge_pair[0],0])
            y0=int(kpnts[edge_pair[0],1])
            x1=int(kpnts[edge_pair[1],0])
            y1=int(kpnts[edge_pair[1],1])
            cv2.line(_img,(x0,y0),(x1,y1),color,3)
    for pnt in kpnts:
        if pnt[2]>0.1:
            cv2.circle(_img,(int(pnt[0]),int(pnt[1])),6,(0,0,255),-1)
    return _img


            

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
cam_num = 3
#frame_num = 0
MAIN_DB_FOLDER = "/home/quinoa/database_mkv"


Model = load_wrapped_model("data/model_t/myModel_MKV_WS_BL.keras",compile=False) 
Model.trainable = False

for frame_num in range(0,4000,10):
    for cam_num in range(4):
        cam_path = os.path.join(MAIN_DB_FOLDER,f"Cam_{cam_num}")
        rgb_path = os.path.join(cam_path,"RGB")
        depth_path = os.path.join(cam_path,"Depth")
        landmarks_path = os.path.join(cam_path,"Landmarks")
        rgb_img = cv2.imread(os.path.join(rgb_path,"rgb_{:06d}.jpg".format(frame_num)))
        depth_img = cv2.imread(os.path.join(depth_path,"depth_{:06d}.png".format(frame_num)))
        depth = tf.expand_dims(tf_3Uint8_to_float32(depth_img[:,:,::-1]),axis=-1)
        lm_data,vis = read_landmarks(os.path.join(landmarks_path,"lm_{:06d}.csv".format(frame_num)))
        #for k,v in enumerate(vis):

        #visi = tf.convert_to_tensor([1.0]*18,dtype=tf.float32)
        print(lm_data,vis)
        print(lm_data.shape,depth.shape)
        squared_rgbd = tf_test_map_affine_woaugment_RGBD(depth,depth.shape,lm_data,vis,njoints=18)
        obbox = tf.cast(squared_rgbd[2][0],tf.float32)
        print(lm_data.shape,squared_rgbd[1][0].shape)
        _obbox = obbox[0:2,0:2]
        padding = obbox[2,0:2]
        print(obbox)
        tensor = squared_rgbd[0] #tf.cast(tf.expand_dims(squared_rgbd[0],axis=0),dtype=tf.dtypes.float32)
        hms = Model.predict(tensor)
        preds = hms[0,1,:,:,:]
        #cv2.imshow(f"results_{cover}",imgs[cover])
        img_bgr = draw_pose(rgb_img,preds,_obbox,padding)
        print(squared_rgbd[1][0])
        img_gt = draw_poseGT(rgb_img,squared_rgbd[1][0])
        plt.imshow(img_bgr[:,:,::-1])#[:,:,::-1])
        plt.figure()
        plt.imshow(img_gt[:,:,::-1])
        plt.figure()
        plt.imshow(depth[:,:,0],cmap="jet")
        #plt.savefig(f"/home/quinoa/sub_{subject_id}-num_{img_num}-{cover}.png", bbox_inches='tight')
        plt.show()
#cv2.destroyAllWindows()