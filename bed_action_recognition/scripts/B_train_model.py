import os,sys
sys.path.append(os.getcwd())
from bed_action_recognition.handlers import BedARHTManager

MAIN_FOLDER = "/home/quinoa/Desktop/dccv_act_recog_sequences"
JSON_FNAME = "bed_action_recognition/data/data_bed_action.ignore.json"
POSE_MODEL_PATH = "data/model_t/myModel_SLP_WS_BL_WithAtt_Depth4C_FT2.keras"

if __name__ == "__main__":
    manager = BedARHTManager(MAIN_FOLDER,
                             JSON_FNAME,
                             train_epochs=40,
                             train_batch_size=20,
                             sequence_length=7,
                             pose_model_path=POSE_MODEL_PATH,
                             action_num=9,
                             pose_njoints=14,
                             pose_emb_size=256)
    manager.train()