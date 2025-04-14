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
import glob
import scipy.io
from numpy import ndarray
from loguru import logger
from pydantic import BaseModel
from pydantic import ValidationError
from scipy.io.matlab._mio5_params import mat_struct as MatStruct
import os
import csv 

class MHADAnnoPoint(BaseModel):
    x: int
    y: int
    id: int
    is_visible:int

def open_mat_file(filename: str) -> Dict:
    """Open a Matlab `.mat` file

    Args:
        filename (str): Matlab file name

    Returns:
        Dict: Matlab object as dictionary
    """
    return scipy.io.loadmat(filename, struct_as_record=False)
    
def read_landmark_data(csv_path: str):
    with open(csv_path,"r") as csv_file:
        reader = csv.reader(csv_file)
        annopoints = [
            MHADAnnoPoint(
                x=int(drow[0]),
                y=int(drow[1]),
                id=idrow,
                is_visible=int(1),#1
                multisubject=0 
            )
            for idrow,drow in enumerate(reader)
        ]
        if len(annopoints)<20:
            print(p.id for p in annopoints)
            raise Exception("CSV reader stopped at 0.0")
    return annopoints

def read_mhad_folder_to_htf_data(
        main_folder)-> Union[List[HTFPersonDatapointRGBD], Tuple[List[HTFPersonDatapointRGBD], Tuple]]:
    record_to_return = []
    for sub_num in range(1,9):
        sub_path =  "{:04d}".format(sub_num)
        for action_num in range(1,28):
            action_path =  os.path.join(sub_path,"{:04d}".format(action_num))
            for trial_num in range(1,2):
                trial_path =  os.path.join(action_path,"{:03d}".format(trial_num))
                for fname in glob.glob(os.path.join(main_folder,trial_path,"Depth")+"/*.png"):
                    depth_path = fname
                    fid = fname.split("/")[-1].split("_")[-1].split(".")[0]
                    annopoints = read_landmark_data(os.path.join(main_folder,trial_path,"LM_Data","lm_{}.csv".format(fid)))
                    if len(annopoints)<20:
                        print(p.id for p in annopoints)
                        raise Exception("CSV reader stopped at 0.0")
                    record_to_return.append(
                    HTFPersonDatapointRGBD(
                        is_train=1,
                        image_id=222,
                        cover="uncover",
                        source_image_rgb = os.path.join(trial_path,"Depth","depth_{}.png".format(fid)), # just removed the main_folder from the path so that it can be prefixed
                        source_image_depth = os.path.join(trial_path,"Depth","depth_{}.png".format(fid)),
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
                            if isinstance(joint, MHADAnnoPoint)
                        ],
                        scale=1.0,
                        multisubject=0,
                        )
                    )
    return record_to_return
    #"""