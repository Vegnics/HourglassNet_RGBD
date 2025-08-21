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

def draw_pose_mplib(depth,hms,use_hms=True,kpnts_loc=None):
    fig,ax = plt.subplots()
    #scalimg = ax.imshow(depth)
    #scalimg = ax.imshow(depth,cmap="jet",vmin=0,vmax=5.5)
    scalimg = ax.imshow(depth,cmap="jet")
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
    
    #fig.colorbar(scalimg,location="left",orientation="vertical",cmap="jet")
    plt.show()

def draw_pose_mplib_new(depth,hms,use_hms=True,kpnts_loc=None,npoints=14):
    fig,ax = plt.subplots()
    #scalimg = ax.imshow(depth)
    scalimg = ax.imshow(depth,cmap="jet",vmin=0,vmax=5.0)
    #scalimg = ax.imshow(depth,cmap="jet")
    if use_hms:
        kpnts = []
        for i in range(npoints):
            pnt = np.argmax(hms[:,:,i])
            x = int((pnt%64))
            y = int((pnt//64))
            val = hms[int(pnt//64),int(pnt%64),i]
            kpnts.append([x,y,val])
        kpnts = np.array(kpnts)
    else:
        kpnts = kpnts_loc.copy()
    _visible_kpts = np.array([i for i in range(npoints)])
    _visible_kpts = list(_visible_kpts[kpnts[:,2]>0.45])
    """
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
    """
    
    """
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
    """

    # 20-Joint scheme (Kinect V1 - MHAD)
    KEYPOINT_EDGE_INDS_TO_COLOR = {
        # Spine
        (0, 1): (255, 0, 0),      # Head <-> Shoulder Center
        (1, 2): (255, 0, 0),      # Shoulder Center <-> Spine
        (2, 3): (255, 0, 0),      # Spine <-> Hip Center

        # Left Arm
        (1, 4): (0, 255, 0),      # Shoulder Center <-> Shoulder Left
        (4, 5): (0, 255, 0),      # Shoulder Left <-> Elbow Left
        (5, 6): (0, 255, 0),      # Elbow Left <-> Wrist Left
        (6, 7): (0, 255, 0),      # Wrist Left <-> Hand Left

        # Right Arm
        (1, 8): (0, 0, 255),      # Shoulder Center <-> Shoulder Right
        (8, 9): (0, 0, 255),      # Shoulder Right <-> Elbow Right
        (9, 10): (0, 0, 255),     # Elbow Right <-> Wrist Right
        (10, 11): (0, 0, 255),    # Wrist Right <-> Hand Right

        # Left Leg
        (3, 12): (255, 255, 0),   # Hip Center <-> Hip Left
        (12, 13): (255, 255, 0),  # Hip Left <-> Knee Left
        (13, 14): (255, 255, 0),  # Knee Left <-> Ankle Left
        (14, 15): (255, 255, 0),  # Ankle Left <-> Foot Left

        # Right Leg
        (3, 16): (255, 0, 255),   # Hip Center <-> Hip Right
        (16, 17): (255, 0, 255),  # Hip Right <-> Knee Right
        (17, 18): (255, 0, 255),  # Knee Right <-> Ankle Right
        (18, 19): (255, 0, 255)   # Ankle Right <-> Foot Right
    }


    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        _color = (color[2]/255.0,color[1]/255.0,color[0]/255.0)
        print(edge_pair)
        if edge_pair[0] in _visible_kpts and edge_pair[1] in _visible_kpts:
            x0=kpnts[edge_pair[0],0]
            y0=kpnts[edge_pair[0],1]
            x1=kpnts[edge_pair[1],0]
            y1=kpnts[edge_pair[1],1]
            pline = mppatches.Polygon(4.0*np.array([[x0,y0],[x1,y1]]),closed=False,color=_color,lw=2.5)
            ax.add_patch(pline)
    for pnt in kpnts:
        if pnt[2]>0.45:
            pcircle = mppatches.Circle((4.0*float(pnt[0]),4.0*float(pnt[1])),color=(1.0,0.0,0.0),radius=4.0)
            ax.add_patch(pcircle)
    #pcircle = mppatches.Circle((10.0,10.0),4.0) 
    #ax.add_patch(pcircle)
    
    #fig.colorbar(scalimg,location="left",orientation="vertical",cmap="jet")
    plt.show()