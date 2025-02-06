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

class SLPAnnoPoint(BaseModel):
    x: int
    y: int
    id: int
    is_visible:int

def read_landmark_data(csv_path: str):
    with open(csv_path,"r") as csv_file:
        reader = csv.reader(csv_file)
        annopoints = [
            SLPAnnoPoint(
                x=int(drow[0]),
                y=int(drow[1]),
                id=idrow,
                is_visible=int(drow[2])#1 
            )
            for idrow,drow in enumerate(reader)
        ]
    return annopoints

def read_slp_folder_to_htf_data(
        main_folder
)-> Union[List[HTFPersonDatapointRGBD], Tuple[List[HTFPersonDatapointRGBD], Tuple]]:
    record_to_return = []
    for sub_num in range(1,91):
        sub_id = "{:05d}".format(sub_num)
        for sample_num in range(1,46):
            sample_id = "{0:06d}".format(sample_num)
            annopoints = read_landmark_data(os.path.join(main_folder,sub_id,"LMData",f"lm_{sample_id}.csv")) 
            for cover_opt in ["uncover","cover1","cover2"]:
                record_to_return.append(
                HTFPersonDatapointRGBD(
                    is_train=1,
                    image_id=sample_num,
                    cover=cover_opt,
                    source_image_rgb=os.path.join(sub_id,"RGB",cover_opt,f"image_{sample_id}.jpg"), # just removed the main_folder from the path so that it can be prefixed
                    source_image_depth = os.path.join(sub_id,"Depth",cover_opt,f"depth_{sample_id}.png"),
                    person_id=sub_num,
                    bbox=HTFPersonBBox(
                        top_left=HTFPoint(x=0, y=0),
                        bottom_right=HTFPoint(x=0, y=0),
                    ),
                    joints=[
                        HTFPersonJoint(
                            x=joint.x, y=joint.y, id=joint.id, visible=bool(joint.is_visible)
                        )
                        for joint in annopoints
                        if isinstance(joint, SLPAnnoPoint)
                    ],
                    scale=1.0,
                    multisubject=0,
                    )
                )
    return record_to_return
        