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
modality = "Depth"
dataset_path = "/home/quinoa/Desktop/some_shit/patient_project/SLP_RGBD_v3"
annotations_path = "data/htf_slp_dataset.ignore.json"

#dataset_path = "/home/quinoa/Desktop/DCCV_bedpose"
#annotations_path = "data/htf_dccv_dataset.ignore.json"


# Related to the ground truth data
n1joints = 14
n2joints = 12
use2joints = False
heatmap_stddev =  1.1
stddev_factor = 1.3
limbs_2J = [(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8),(8,12),(9,12),(12,13),(11,10),(10,9)] 
enable_visibility = False

# Related to the model and training
model_name = "myModel_SLP_WS_BL_1B_ATT9FTTEST_2_Depth4C.keras" #"myModel_SLP_WS_BL_1B_w2joints.keras"
chkpnt_path = os.path.join("data/model_t",model_name)
csv_logger = f"logs/myModelLogs_{model_name}_{current_time}.csv"
epochs = 150
batch_size = 20
stages = 2
stage_filters = 256
residual_nblocks = 1
learning_rate = 1.5e-6

# Related to loading pre-trained models
load_model = True
pt_model_name = "data/model_t/myModel_SLP_WS_BL_1B_ATT9FTTEST_Depth4C.keras" #"/home/quinoa/Documents/models_slp/myModel_SLP_BL_1B_HG1Depth4C.keras" #"data/model_t/myModel_SLP_WS_BL_1B_FT2_Depth4C.keras" # # #"data/model_t/myModel_SLP_WS_BL_1B_ATT9FT_Depth4C.keras" #
loading_mode =  "Partial_Train_F2S" #"Partial_Train_Attention" #"Partial_Downsampling_frozen" # # # # # #"Partial_Downsampling_frozen" # #"Full_Train" #"Partial_Downsampling_frozen"

#Related to the attention mechanisms
skip_AM = "NoAM"
s2f_AM =  "FAM" 
f2s_AM = "SAM" 

#Related to the loss
W_1jnts = 1.0
W_2jnts = 0.2 
W_coords = 0.000000




# Related
limbs_2J_str = [f"{limb}" for limb in limbs_2J]
conf["data"]["input"]["mode"] = modality
conf["data"]["input"]["source"] = dataset_path
conf["data"]["output"]["source"] = annotations_path
conf["data"]["output"]["joints"]["num"] = n1joints

conf["dataset"]["heatmap"]["channels_1JHMs"] = n1joints
conf["dataset"]["heatmap"]["channels_2JHMs"] = n2joints
conf["dataset"]["heatmap"]["stddev"] = heatmap_stddev
conf["dataset"]["heatmap"]["stddev_factor"] = stddev_factor
conf["dataset"]["heatmap"]["stacks"] = stages
conf["dataset"]["heatmap"]["limbs_2J_str"] = limbs_2J_str
conf["dataset"]["heatmap"]["enable_visibility"] = enable_visibility
conf["dataset"]["data_mode"] = modality

conf["model"]["params"]["channels_1joint"] = n1joints
conf["model"]["params"]["channels_2joint"] = n2joints
conf["model"]["params"]["use_2jointHM"] = use2joints
conf["model"]["params"]["skip_AM"] = skip_AM
conf["model"]["params"]["s2f_AM"] = s2f_AM
conf["model"]["params"]["f2s_AM"] = f2s_AM
conf["model"]["params"]["stages"] = stages
conf["model"]["params"]["stage_filters"] = stage_filters
conf["model"]["params"]["residual_nblocks"] = residual_nblocks
conf["model"]["load_model"] = load_model
conf["model"]["model_path"] = pt_model_name
conf["model"]["loading_style"] = loading_mode
conf["model"]["params"]["channel_number"] = 4


conf["train"]["epochs"] = epochs
conf["train"]["batch_size"] = batch_size
conf["train"]["learning_rate"] = learning_rate

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