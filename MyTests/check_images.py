import os
import numpy as np
import cv2

main_folder = "/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD"
for sub_num in range(1,103):
        sub_id = "{:05d}".format(sub_num)
        for sample_num in range(1,2):
            sample_id = "{0:06d}".format(sample_num)
            #annopoints = read_landmark_data(os.path.join(main_folder,sub_id,"LMData",f"lm_{sample_id}.csv")) 
            for cover_opt in ["uncover"]:
                source_image_rgb= os.path.join(main_folder,sub_id,"RGB",cover_opt,f"image_{sample_id}.jpg") # just removed the main_folder from the path so that it can be prefixed
                print(source_image_rgb)
                img = cv2.imread(source_image_rgb)
                cv2.imshow("Image",img)
                cv2.waitKey(0)   
cv2.destroyAllWindows()