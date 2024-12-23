import json
from typing import Dict
from typing import List

from loguru import logger

import sys,os
#print(os.getcwd())
sys.path.insert(1,os.getcwd())
#print(sys.path)

from hourglass_tensorflow.utils.parsers import MPIIDatapoint
from hourglass_tensorflow.utils.parsers import parse_mpii
from hourglass_tensorflow.utils.writers import common_write
from hourglass_tensorflow.utils.parsers.slp import read_slp_folder_to_htf_data

#MAT_FILE = "data/mpii.ignore.mat"
#MAT_FILE = "data/mpii_human_pose.mat"\
HTF_JSON = "data/htf_slp_test.ignore.json"
SLP_FOLDER = "/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD"

if __name__ == "__main__":
    # Parse file as list of records
    logger.info("Convert from MPII to HTF")
    htf_data = read_slp_folder_to_htf_data(SLP_FOLDER)
    # Write Transform data
    logger.info(f"Write HTF data to {HTF_JSON}")
    common_write(htf_data, HTF_JSON)
