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

from hourglass_tensorflow.types import HTFPersonDatapointRGBD

#class HTFDBpoints(BaseModel):
#    data: List[HTFPersonDatapoint]

#HTF_JSON = "/content/HourglassNet_RGBD/data/htf_slp.ignore.json"
#HTF_DATASET_JSON = "/content/HourglassNet_RGBD/data/htf_slp_dataset.ignore.json"

HTF_JSON = "data/htf_slp.ignore.json"
HTF_DATASET_JSON = "data/htf_slp_dataset.ignore.json"

if __name__ == "__main__":
    # Parse file as list of records
    logger.info(f"Reading HTF data at {HTF_JSON}")
    #data = parse_file_as(List[HTFPersonDatapoint], HTF_JSON)
    with open(HTF_JSON) as file:
        DataJson = json.load(file)
    data = [HTFPersonDatapointRGBD.model_validate(datap) for datap in DataJson]
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
    #Forced joint IDs
    forced_ids = [1,2,3,4,6,7,8,9,12,13]
    print(avg_joints_per_sample,avg_visible_joints_per_sample)
    # Prepare data as table
    DATA = []
    for datap in data:
        cntVis = 0
        jids = [j.id for j in datap.joints]
        _jids = set(jids)
        jxs = [j.x for j in datap.joints]
        jys = [j.y for j in datap.joints]
        jvis = [j.visible for j in datap.joints]
        if len(_jids)<14:
            print(jids)
            raise Exception("CSV reader stopped at 0.0")
        d = {"set": "TRAIN" if datap.is_train else "VALIDATION",
        "image": datap.source_image_rgb,
        "depth": datap.source_image_depth,
        "cover": datap.cover,
        "scale":datap.scale,
        "multisubject":datap.multisubject,
        "bbox_tl_x": datap.bbox.top_left.x,
        "bbox_tl_y": datap.bbox.top_left.y,
        "bbox_br_x": datap.bbox.bottom_right.x,
        "bbox_br_y": datap.bbox.bottom_right.y,
        "center_x": -1,
        "center_y": -1,
        }
        for jid in range(14):
            if jid in jids :#and jid in forced_ids:
                k = jids.index(jid)
                d[f"joint_{jid}_X"] = jxs[k]
                d[f"joint_{jid}_Y"] = jys[k]
                #if jvis[k]==0 and False:
                #    d[f"joint_{jid}_X"] = -100000
                #    d[f"joint_{jid}_Y"] = -100000
                #else:
                #    d[f"joint_{jid}_X"] = jxs[k]
                #    d[f"joint_{jid}_Y"] = jys[k]
                d[f"joint_{jid}_visible"] = jvis[k] #True 
        DATA.append(d)
    # Write Transformed data
    with open(HTF_DATASET_JSON,"w") as file:
        json.dump(DATA,file)