import yaml
from datetime import datetime
import os

cfg_name = "config/train_rgbd.default.yaml"
outcfg_name = "config/train_rgbd_gen.yaml"

with open(cfg_name, "r") as f:
    conf = yaml.safe_load(f)

# Get current time
current_time = datetime.now().strftime("%H:%M:%S")

# Related to the data and dataset
modality = "RGBD"
dataset_path = "/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD_v3"
annotations_path = "data/htf_slp_dataset.ignore.json"

# Related to the ground truth data
n1joints = 14
n2joints = 12
use2joints = True
heatmap_stddev =  1.1
stddev_factor = 1.3

# Related to the model and training
model_name = "myModel_SLP_fAB10_2j"
chkpnt_path = os.path.join("data/model_t",model_name)
csv_logger = f"logs/myModelLogs_{model_name}_{current_time}.csv"
epochs = 200
batch_size = 20
stages = 2

#Related to the attention mechanisms
skip_AM = "NoAM"
s2f_AM =  "NoAM" 
f2s_AM = "NoAM" 

#Related to the loss
W_1jnts = 1.0
W_2jnts = 0.0 
W_coords = 0.0



# Related 
conf["data"]["input"]["mode"] = modality
conf["data"]["input"]["source"] = dataset_path
conf["data"]["output"]["source"] = annotations_path
conf["data"]["output"]["joints"]["num"] = n1joints

conf["dataset"]["heatmap"]["channels"] = n1joints
conf["dataset"]["heatmap"]["stddev"] = heatmap_stddev
conf["dataset"]["heatmap"]["stddev_factor"] = stddev_factor
conf["dataset"]["heatmap"]["stacks"] = stages

conf["model"]["params"]["channels_1joint"] = n1joints
conf["model"]["params"]["channels_2joint"] = n2joints
conf["model"]["params"]["use_2jointHM"] = use2joints
conf["model"]["params"]["skip_AM"] = skip_AM
conf["model"]["params"]["s2f_AM"] = s2f_AM
conf["model"]["params"]["f2s_AM"] = f2s_AM
conf["model"]["params"]["stages"] = stages


conf["train"]["epochs"] = epochs
conf["train"]["batch_size"] = batch_size

conf["train"]["loss"]["params"]["WL2_j1"] =  W_1jnts
conf["train"]["loss"]["params"]["WL2_j2"] =  W_2jnts * float(use2joints)
conf["train"]["loss"]["params"]["WCoords"] = W_coords
conf["train"]["loss"]["params"]["nstages"] = stages
conf["train"]["loss"]["params"]["n1joints"] = n1joints
conf["train"]["loss"]["params"]["n2joints"] = n2joints
conf["train"]["loss"]["params"]["use2joints"] = use2joints

conf["train"]["metrics"][0]["params"]["num_1joints"] = n1joints
conf["train"]["metrics"][1]["params"]["num_1joints"] = n1joints

conf["train"]["callbacks"][1]["params"]["filepath"] = chkpnt_path
conf["train"]["callbacks"][3]["params"]["filename"] = csv_logger


with open(outcfg_name, "w") as f:
    yaml.safe_dump(conf,f,indent=2,sort_keys=False)