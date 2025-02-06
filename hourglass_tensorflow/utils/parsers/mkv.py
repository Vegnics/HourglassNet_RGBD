from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Tuple
from typing import Iterable
from typing import Optional

from hourglass_tensorflow.types import HTFPoint
from hourglass_tensorflow.types import HTFPersonBBox
from hourglass_tensorflow.types import HTFPersonJoint
from hourglass_tensorflow.types import HTFPersonDatapointRGBD

import scipy.io
from numpy import ndarray
from loguru import logger
from pydantic import BaseModel
from pydantic import ValidationError
from scipy.io.matlab._mio5_params import mat_struct as MatStruct
import os
import csv 

class MKVAnnoPoint(BaseModel):
    x: int
    y: int
    id: int
    is_visible:int
    multisubject: int
    
def read_landmark_data(csv_path: str):
    with open(csv_path,"r") as csv_file:
        reader = csv.reader(csv_file)
        annopoints = [
            MKVAnnoPoint(
                x=int(drow[0]),
                y=int(drow[1]),
                id=idrow,
                is_visible=int(drow[2]),#1
                multisubject=int(drow[3]) 
            )
            for idrow,drow in enumerate(reader)
        ]
        if len(annopoints)<17:
            print(p.id for p in annopoints)
            raise Exception("CSV reader stopped at 0.0")
    return annopoints

def read_mkv_folder_to_htf_data(
        main_folder,frame_start,frame_end
)-> Union[List[HTFPersonDatapointRGBD], Tuple[List[HTFPersonDatapointRGBD], Tuple]]:
    record_to_return = []
    for cam_num in range(4):
        #cam_path = os.path.join(main_folder,f"Cam_{cam_num}")
        cam_path = f"Cam_{cam_num}"
        rgb_path = os.path.join(cam_path,"RGB")
        depth_path = os.path.join(cam_path,"Depth")
        landmarks_path = os.path.join(main_folder,cam_path,"Landmarks")
        for sample_num in range(frame_start,frame_end):
            annopoints = read_landmark_data(os.path.join(landmarks_path,"lm_{:06d}.csv".format(sample_num)))
            if len(annopoints)<17:
                if len(annopoints)<17:
                    print(p.id for p in annopoints)
                    raise Exception("CSV reader stopped at 0.0")
            record_to_return.append(
            HTFPersonDatapointRGBD(
                is_train=1,
                image_id=sample_num,
                cover="uncover",
                source_image_rgb=os.path.join(rgb_path,"rgb_{:06d}.jpg".format(sample_num)), # just removed the main_folder from the path so that it can be prefixed
                source_image_depth = os.path.join(depth_path,"depth_{:06d}.png".format(sample_num)),
                person_id=1,
                bbox=HTFPersonBBox(
                    top_left=HTFPoint(x=0, y=0),
                    bottom_right=HTFPoint(x=0, y=0),
                ),
                joints=[
                    HTFPersonJoint(
                        x=joint.x, y=joint.y, id=joint.id, visible=bool(joint.is_visible)
                    )
                    for joint in annopoints
                    if isinstance(joint, MKVAnnoPoint)
                ],
                scale=1.0,
                multisubject=annopoints[0].multisubject,
                )
            )
    return record_to_return
        