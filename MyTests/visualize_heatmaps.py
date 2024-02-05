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


Model = tf.keras.models.load_model("data/model_t/myModel_b",
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

img_bgr = cv2.imread("data/test_tennis2.png")
img = tf_load_image("data/test_tennis2.png")
tensor = tf.cast(tf.expand_dims(img,axis=0),dtype=tf.dtypes.float32)
#print(tensor)
hms = Model.predict(tensor)
preds = hms[0,-1,:,:,:]
for i in range(16):
    hm = preds[:,:,i]
    pnt = np.argmax(hm)
    x = int(4*(pnt%64))
    y = int(4*(pnt//64))
    #cv2.circle(img_bgr,(x,y),5,(0,0,255),-1)
    print(f"Landmark {LM_NAMES[i]}:  ({x},{y})")
    plt.imshow(hm,cmap="jet")
    plt.show()
#plt.imshow(img_bgr[:,:,::-1])
#plt.show()

