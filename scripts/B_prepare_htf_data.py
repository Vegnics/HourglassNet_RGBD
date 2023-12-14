from typing import Dict
from typing import List

import numpy as np
from loguru import logger
#from pydantic import parse_file_as
from pydantic import BaseModel,TypeAdapter
import json
import pandas as pd

import sys,os
sys.path.insert(1,os.getcwd())

from hourglass_tensorflow.types import HTFPersonDatapoint

class HTFDBpoints(BaseModel):
    data: List[HTFPersonDatapoint]

HTF_JSON = "data/htf.ignore.json"
HTF_DATASET_JSON = "data/htf_dataset.ignore.json"

if __name__ == "__main__":
    # Parse file as list of records
    logger.info(f"Reading HTF data at {HTF_JSON}")
    #data = parse_file_as(List[HTFPersonDatapoint], HTF_JSON)
    with open(HTF_JSON) as file:
        DataJson = json.load(file)
    data = [HTFPersonDatapoint.model_validate(datap) for datap in DataJson]
    # Compute Stats
    ## Average number of joints and average number of visible joints
    # d is a HTFPersonDataPoint
    num_joints = [len(d.joints) for d in data]
    num_visible_joints = [len([j for j in d.joints if j.visible]) for d in data]
    avg_joints_per_sample = np.mean(num_joints)
    avg_visible_joints_per_sample = np.mean(num_visible_joints)
    ## Joint ID distribution
    joints_id = [(j.id, j.visible) for d in data for j in d.joints]
    only_visible_joints_id = [jid for jid, j_visible in joints_id if j_visible]
    print(avg_joints_per_sample,avg_visible_joints_per_sample)
    # Prepare data as table
    DATA = []
    for datap in data:
        if len(datap.joints)==16:
            d = {"set": "TRAIN" if datap.is_train else "VALIDATION",
            "image": datap.source_image,
            "scale":datap.scale,
            "bbox_tl_x": datap.bbox.top_left.x,
            "bbox_tl_y": datap.bbox.top_left.y,
            "bbox_br_x": datap.bbox.bottom_right.x,
            "bbox_br_y": datap.bbox.bottom_right.y,
            "center_x": datap.center.x,
            "center_y": datap.center.y,
            }
            for j in datap.joints:
                d[f"joint_{j.id}_X"] = j.x
                d[f"joint_{j.id}_Y"] = j.y
                d[f"joint_{j.id}_visible"] = j.visible
            DATA.append(d)
    # Write Transformed data
    with open(HTF_DATASET_JSON,"w") as file:
        json.dump(DATA,file)