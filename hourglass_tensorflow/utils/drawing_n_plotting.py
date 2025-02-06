import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as mppatches
from hourglass_tensorflow.utils.tf import tf_batch_matrix_argmax

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


def draw_pose_HG(img,hm):
    _img = np.copy(img)
    kpnts = []
    for i in range(14):
        pnt = np.argmax(hm[:,:,i])
        x = int((pnt%64))
        y = int((pnt//64))
        #dx,dy = get_secondmax(hm[:,:,i],x,y)
        #x = 4*int((pnt%64) + 0.5*dx)
        #y = 4*int((pnt//64) + 0.5*dy)
        #x = int((N/64.0)*((pnt%64) + 0.25*dx - padx)+ obbox[0,0])
        #y = int((N/64.0)*((pnt//64) + 0.25*dy - pady)+ obbox[0,1])
        val = hm[int(pnt//64),int(pnt%64),i]
        #print(f"Landmark {LM_NAMES[i]}:  ({x},{y},{val})")
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

def draw_pose_mplib(depth,hms):
    fig,ax = plt.subplots()
    scalimg = ax.imshow(depth,cmap="jet",vmin=0.0,vmax=9.0)
    #scalimg = ax.imshow(depth,cmap="jet")
    kpnts = []
    for i in range(14):
        pnt = np.argmax(hms[:,:,i])
        x = int((pnt%64))
        y = int((pnt//64))
        val = hms[int(pnt//64),int(pnt%64),i]
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
        _color = (color[2]/255.0,color[1]/255.0,color[0]/255.0)
        if edge_pair[0] in _visible_kpts and edge_pair[1] in _visible_kpts:
            x0=int(kpnts[edge_pair[0],0])
            y0=int(kpnts[edge_pair[0],1])
            x1=int(kpnts[edge_pair[1],0])
            y1=int(kpnts[edge_pair[1],1])
            pline = mppatches.Polygon(4.0*np.array([[x0,y0],[x1,y1]]),closed=False,color=_color,lw=2.5)
            ax.add_patch(pline)
    for pnt in kpnts:
        if pnt[2]>0.45:
            pcircle = mppatches.Circle((4.0*float(pnt[0]),4.0*float(pnt[1])),color=(1.0,0.0,0.0),radius=4.0)
            ax.add_patch(pcircle)
    #pcircle = mppatches.Circle((10.0,10.0),4.0) 
    #ax.add_patch(pcircle)
    fig.colorbar(scalimg,location="left",orientation="vertical",cmap="jet")
    plt.show()