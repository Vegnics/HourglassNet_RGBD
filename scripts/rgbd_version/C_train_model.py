from typing import Dict
from typing import List

import numpy as np
from loguru import logger
#from pydantic import parse_file_as

import sys,os
sys.path.insert(1,os.getcwd())

from hourglass_tensorflow.handlers import HTFManager

CONFIG_FILE = "config/train_rgbd.default.yaml"

if __name__ == "__main__":
    # Parse file as list of records
    logger.info(f"Reading HTF data at {CONFIG_FILE}")
    print("aomasd")
    manager = HTFManager(filename=CONFIG_FILE, verbose=True)
    manager()
    