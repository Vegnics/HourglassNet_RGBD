import json
from typing import Dict
from typing import List

from loguru import logger

import sys,os
sys.path.insert(1,os.getcwd())

from hourglass_tensorflow.utils.parsers import MPIIDatapoint
from hourglass_tensorflow.utils.parsers import parse_mpii
from hourglass_tensorflow.utils.writers import common_write
from hourglass_tensorflow.utils.parsers.slp import read_slp_folder_to_htf_data
from hourglass_tensorflow.utils.parsers.mkv import read_mkv_folder_to_htf_data

#MAT_FILE = "data/mpii.ignore.mat"
#MAT_FILE = "data/mpii_human_pose.mat"\

HTF_JSON = "/content/HourglassNet_RGBD/data/htf_slp.ignore.json"
HTF_JSON = "/content/HourglassNet_RGBD/data/htf_mkv.ignore.json"
SLP_FOLDER = "/content/SLP_RGBD_v2"


#HTF_JSON = "data/htf_slp_test.ignore.json"
HTF_JSON = "data/htf_slp.ignore.json"

#HTF_JSON = "data/htf_slp.ignore.json"

SLP_FOLDER = "/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD_v3"
MKV_FOLDER = "/home/quinoa/database_mkv"

if __name__ == "__main__":
    # Parse file as list of records
    logger.info("Convert from MKV-RGBD to HTF")
    htf_data = read_slp_folder_to_htf_data(SLP_FOLDER)
    #htf_data = read_mkv_folder_to_htf_data(MKV_FOLDER,0,4714)
    # Write Transform data
    logger.info(f"Write HTF data to {HTF_JSON}")
    common_write(htf_data, HTF_JSON)
