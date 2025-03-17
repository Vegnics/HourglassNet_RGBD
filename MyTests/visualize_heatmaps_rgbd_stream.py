import tensorflow as tf
import keras
import sys,os
import sys,os
from matplotlib import pyplot as plt
sys.path.insert(1,os.getcwd())
import numpy as np
import cv2 
import csv
import json

from hourglass_tensorflow.metrics.correct_keypoints import *
from hourglass_tensorflow.losses.mae_custom import *
from hourglass_tensorflow.utils.tf import tf_load_image,tf_3Uint8_to_float32
from hourglass_tensorflow.handlers._transformation import tf_train_map_squarify,tf_test_map_affine_woaugment_RGBD,tf_test_map_squarify
from hourglass_tensorflow.metrics.distance import OverallMeanDistance,SoftargmaxMeanDist
from hourglass_tensorflow.models import HourglassModel

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
    if hm[y+dy,x+dx]>0.1:
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
        x = int((N/64.0)*((pnt%64) + 0.0*dx - padx)+ obbox[0,0])
        y = int((N/64.0)*((pnt//64) + 0.0*dy - pady)+ obbox[0,1])
        val = hm[int(pnt//64),int(pnt%64),i]
        print(f"Landmark {LM_NAMES[i]}:  ({x},{y},{val})")
        kpnts.append([x,y,val])
    kpnts = np.array(kpnts)
    _visible_kpts = np.array([i for i in range(14)])
    _visible_kpts = list(_visible_kpts[kpnts[:,2]>0.1])
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
        if pnt[2]>0.1:
            cv2.circle(_img,(int(pnt[0]),int(pnt[1])),9,(0,0,255),-1)
    return _img

def load_bboxes(fpath):
    with open(os.path.join(fpath, "bboxes_info.json"), "r") as file:
        jdata = json.load(file)
        return jdata["bboxes"]
    
Model = tf.keras.models.load_model("data/model_t/myModel_SLP_fAB10_2j",
                           custom_objects= {#"RatioCorrectKeypoints":RatioCorrectKeypoints
                                            "HourglassModel": HourglassModel,
                                            "PercentageOfCorrectKeypoints":PercentageOfCorrectKeypoints,
                                            "MAE_custom":MAE_custom,
                                            "OverallMeanDistance":OverallMeanDistance,
                                            "SoftargmaxMeanDist":SoftargmaxMeanDist},compile=False)
Model.trainable = False
print(Model)
print(Model.get_config())
Model.summary()
            

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

MAIN_FOLDER = "/home/quinoa/Desktop/ntu_patient_action_recognition"
subject_num = 0
sample_num = 2
action_num = 0

rgb_fname = os.path.join(MAIN_FOLDER,"{:04d}/{:03d}/{:04d}".format(subject_num,action_num,sample_num),"RGB","rgb_{:04d}.mp4".format(sample_num))
bboxes = []
_cnt = 0 
cap = cv2.VideoCapture(rgb_fname)
fbboxpath = os.path.join(MAIN_FOLDER,"{:04d}/{:03d}/{:04d}".format(subject_num,action_num,sample_num)) 
bboxes = load_bboxes(fbboxpath)
k = 0
cbbox = [(0,0),(0,0)]
rgb_frame_buffer = []
while cap.isOpened():
    ret,frame = cap.read()
    if ret:
        cbbox = bboxes[k]
        rframe = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.rectangle(rframe,cbbox[0],cbbox[1],(0,0,255),5)
        imgrgb = rframe[:,:,::-1]
        depth_fname = os.path.join(MAIN_FOLDER,"{:04d}/{:03d}/{:04d}".format(subject_num,action_num,sample_num),"Depth","depth_{:04d}_{:06d}.png".format(sample_num,k))
        imagedepth = cv2.imread(depth_fname)
        imagedepthRot = cv2.rotate(imagedepth,cv2.ROTATE_90_COUNTERCLOCKWISE)
        imagedepth = tf.convert_to_tensor(imagedepthRot[:,:,::-1])
        depthmap = tf.expand_dims(tf_3Uint8_to_float32(imagedepth),axis=2)
        #print(depthmap)
        _depthimg,tbbox = tf_test_map_squarify(depthmap,tf.convert_to_tensor(cbbox))
        #print(_depthimg)
        _obbox = tbbox[0:2,0:2]
        padding = tbbox[2,0:2]
        hms = Model.predict(tf.expand_dims(_depthimg,axis=0))
        preds = hms[0,-1,:,:,:]
        img_bgr = draw_pose(imgrgb,preds,_obbox,padding)
        fig, ax = plt.subplots()
        ax.set_axis_off()  # Hide the axes
        fig.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0.0)  # Remove padding
        ax.margins(0)
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(tf.reshape(_depthimg,(256,256)),cmap="jet")
        # Draw the canvas and get the image as a NumPy array
        fig.canvas.draw()
        # Convert to NumPy array
        depthplot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        depthplot = depthplot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # Close the figure to free memory
        #plt.close(fig)
        #plt.imshow(img_bgr)#[:,:,::-1])
        #plt.figure()
        #plt.imshow(depthplot)#tf.reshape(_depthimg,(256,256)),cmap="jet")
        #plt.show()
        #cv2.rectangle(rframe,cbbox[0],cbbox[1],(0,0,255),5)
        rplotdetph = cv2.resize(depthplot[:,160:-160,:],dsize=(-1,-1),fx=1.0*img_bgr.shape[0]/depthplot.shape[0],fy=1.0*img_bgr.shape[0]/depthplot.shape[0])
        rplotdetph = cv2.resize(rplotdetph,dsize=(rplotdetph.shape[1],img_bgr.shape[0]))
        plotfinal = np.hstack((img_bgr,rplotdetph))
        plotfinal = cv2.resize(plotfinal[:,:,::-1],dsize=(-1,-1),fx=0.8,fy=0.8)
        #print(plotfinal.shape)
        rgb_frame_buffer.append(np.copy(plotfinal))
        cv2.imshow("results",plotfinal)
        cv2.waitKey(0)
        k +=1 
    else:
        break
vwriter = cv2.VideoWriter()
if vwriter.open("/home/quinoa/test_action2.mp4",cv2.VideoWriter_fourcc(*'mp4v'),
                fps=30.0,
                frameSize=(944,768)):  #CHANGE THE FRAME SIZE
    for frame in rgb_frame_buffer:
        vwriter.write(frame)
    #fbboxpath = os.path.join(MAIN_FOLDER,"{:04d}/{:03d}/{:04d}".format(subject_num,action_num,sample_num)) 
    #save_bboxes(bboxes,fbboxpath)
#cv2.destroyAllWindows()
