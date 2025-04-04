import numpy as np
import cv2
import os
import json
import sys
sys.path.insert(1,os.getcwd())
from hourglass_tensorflow.utils.tf import tf_load_image,tf_3Uint8_to_float32
from hourglass_tensorflow.handlers._transformation import tf_test_map_affine_woaugment_RGBD
from hourglass_tensorflow.utils.loaders.model_loader import load_wrapped_model
import tensorflow as tf
import matplotlib.pyplot as plt


keypoint_names = [
    "00_rAnkle",
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

def draw_pose_opencv(src_img,hms,use_hms=True,kpnts_loc=None):
    _draw_img = src_img.copy()
    #fig,ax = plt.subplots()
    #scalimg = ax.imshow(depth,cmap="jet",vmin=0,vmax=5.5)
    #scalimg = ax.imshow(depth,cmap="jet")
    if use_hms:
        kpnts = []
        for i in range(14):
            pnt = np.argmax(hms[:,:,i])
            x = int((pnt%64))
            y = int((pnt//64))
            val = hms[int(pnt//64),int(pnt%64),i]
            kpnts.append([x,y,val])
        kpnts = np.array(kpnts)
    else:
        kpnts = kpnts_loc.copy()
    _visible_kpts = np.array([i for i in range(14)])
    #_visible_kpts = list(_visible_kpts[kpnts[:,2]>0.45])
    #_visible_kpts = [True]*14
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
        #_color = (color[2]/255.0,color[1]/255.0,color[0]/255.0)
        _color = (int(color[0]),int(color[1]),int(color[2]))
        if edge_pair[0] in _visible_kpts and edge_pair[1] in _visible_kpts:
            x0=int(kpnts[edge_pair[0],0])
            y0=int(kpnts[edge_pair[0],1])
            x1=int(kpnts[edge_pair[1],0])
            y1=int(kpnts[edge_pair[1],1])
            cv2.line(_draw_img,(x0,y0),(x1,y1),_color,thickness=5)
            #pline = mppatches.Polygon(4.0*np.array([[x0,y0],[x1,y1]]),closed=False,color=_color,lw=2.5)
            #ax.add_patch(pline)
    for pnt in kpnts:
        x = int(pnt[0])
        y = int(pnt[1])
        cv2.circle(_draw_img,(x,y),7,(0,0,255),-1)
        #if pnt[2]>0.45:
            #pcircle = mppatches.Circle((4.0*float(pnt[0]),4.0*float(pnt[1])),color=(1.0,0.0,0.0),radius=4.0)
            #ax.add_patch(pcircle)
    #pcircle = mppatches.Circle((10.0,10.0),4.0) 
    #ax.add_patch(pcircle)
    #fig.colorbar(scalimg,location="left",orientation="vertical",cmap="jet")
    #plt.show()
    return _draw_img

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
    (0, 1): (235,0,255),
    (1, 2): (235,0,255),
    (2, 3): (80,160,60),
    (3, 4): (0,245,230),
    (4, 5): (0,245,230),
    (6, 7): (235,0,255),
    (7, 8): (235,0,255),
    (8,12): (235,0,255),
    (9,12): (0,245,230),
    (9, 10): (0,245,230),
    (10, 11): (0,245,230),
    (2, 12): (235,0,255),
    (3, 12): (0,245,230),
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
        print(f"Landmark {keypoint_names[i]}:  ({x},{y},{val})")
        kpnts.append([x,y,val])
    kpnts = np.array(kpnts)
    _visible_kpts = np.array([i for i in range(14)])
    _visible_kpts = list(_visible_kpts[kpnts[:,2]>0.13])
    KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (235,0,255),
    (1, 2): (235,0,255),
    (2, 3): (80,160,60),
    (3, 4): (0,245,230),
    (4, 5): (0,245,230),
    (6, 7): (235,0,255),
    (7, 8): (235,0,255),
    (8,12): (235,0,255),
    (9,12): (0,245,230),
    (9, 10): (0,245,230),
    (10, 11): (0,245,230),
    (2, 12): (235,0,255),
    (3, 12): (0,245,230),
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
        if pnt[2]>0.13:
            cv2.circle(_img,(int(pnt[0]),int(pnt[1])),9,(0,0,255),-1)
    return _img
    

def cvt3Uint8toUint32(depth_array):
    _depth_array = np.zeros(depth_array.shape[0:2],dtype=np.float32)
    _depth_array += depth_array[:,:,0]*(256**2)
    _depth_array += depth_array[:,:,1]*256
    _depth_array += depth_array[:,:,2]
    return _depth_array

def save_bboxes(bboxes,fpath):
    jdata = {"bboxes":[]}
    jdata["bboxes"] = bboxes
    with open(os.path.join(fpath, "bboxes_info.json"), "w") as file:
        json.dump(jdata, file)

def load_bboxes(fpath):
    with open(os.path.join(fpath, "bboxes_info.json"), "r") as file:
        jdata = json.load(file)
        return jdata["bboxes"]



MAIN_FOLDER = "/home/quinoa/Desktop/ntu_patient_pose"
with open(os.path.join(MAIN_FOLDER,"db_info.json")) as file:
    dbmeta = json.load(file)
subjects = dbmeta["Subjects"]
subject_num = 3
sample_num = 5
cover = 0 

Model = load_wrapped_model("data/model_t/myModel_SLP_WS_BL_1B_ATT_Depth4C.keras",compile=False) 
Model.trainable = False

for sub in subjects:
    subject_num = sub["subject_id"]
    for sample_num in range(int(sub["sample_counters"])):
        imgs = [0]*3
        key = 0
        with open(os.path.join(MAIN_FOLDER,"{:04d}/{:04d}/landmark_data.json".format(subject_num,sample_num)),"r") as ff:
                lm_data = json.load(ff)
        kpnts_rgb = np.array(lm_data["keypoints"],dtype=np.int32)
        for cover in range(3):
            rgb_fname = os.path.join(MAIN_FOLDER,"{:04d}/{:04d}".format(subject_num,sample_num),"RGB","rgb_{:04d}_C{}.jpg".format(sample_num,cover))
            depth_fname = os.path.join(MAIN_FOLDER,"{:04d}/{:04d}".format(subject_num,sample_num),"Depth","depth_{:04d}_C{}.png".format(sample_num,cover))
            print(rgb_fname)
            rgb_img = cv2.imread(rgb_fname,cv2.IMREAD_COLOR)
            imgs[cover] = rgb_img.copy()
            imgs[cover] = cv2.rotate(imgs[cover],cv2.ROTATE_90_COUNTERCLOCKWISE)
            #imgs[cover] = draw_pose_opencv(imgs[cover],None,use_hms=False,kpnts_loc=kpnts_rgb)
            depth_img = cv2.imread(depth_fname,cv2.IMREAD_COLOR)
            imagedepthRot = cv2.rotate(depth_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
            depth_img = tf.convert_to_tensor(imagedepthRot[:,:,::-1])
            depthmap = tf.expand_dims(tf_3Uint8_to_float32(depth_img),axis=2)
            shape_d = tf.shape(depthmap)
            rgb_zeros = tf.zeros(shape=(shape_d[0],shape_d[1],3),dtype=tf.float32)
            rgbd_image_in = tf.concat([depthmap,rgb_zeros],axis=-1)
            landmarks = tf.convert_to_tensor(lm_data["keypoints"])
            visibilities = tf.convert_to_tensor([1]*14)
            #"""
            axis_rot_mask = tf.convert_to_tensor([1,0,0,0],dtype=tf.float32)
            squared_rgbd = tf_test_map_affine_woaugment_RGBD(rgbd_image_in,rgbd_image_in.shape,landmarks,visibilities,njoints=14,affine_axis_mask=axis_rot_mask)
            obbox = tf.cast(squared_rgbd[2][0],tf.float32)
            print(landmarks.shape,squared_rgbd[1][0].shape)
            _obbox = obbox[0:2,0:2]
            padding = obbox[2,0:2]
            print(obbox)
            tensor = squared_rgbd[0] #tf.cast(tf.expand_dims(squared_rgbd[0],axis=0),dtype=tf.dtypes.float32)
            hms = Model.predict(tensor)
            preds = hms[0,1,:,:,:]
            #cv2.imshow(f"results_{cover}",imgs[cover])
            img_bgr = draw_pose(imgs[cover][:,:,::-1],preds,_obbox,padding)
            #"""
            #img_gt = draw_poseGT(imgs[cover][:,:,::-1],squared_rgbd[1][0])
            img_gt = draw_poseGT(imgs[cover][:,:,::-1],landmarks)
            plt.imshow(img_bgr) #imgs[cover][:,:,::-1])#[:,:,::-1])
            plt.figure()
            plt.imshow(img_gt)
            plt.figure()
            plt.imshow(depthmap[:,:,0],cmap="jet",vmin=800,vmax=6500)
            #plt.figure()
            #plt.imshow(tensor[0,:,:,0],cmap="jet")
            #plt.savefig(f"/home/quinoa/sub_{subject_id}-num_{img_num}-{cover}.png", bbox_inches='tight')
            plt.show()
        #while(key!=ord('q')):
        #    key = cv2.waitKey(0)
#cv2.destroyAllWindows()