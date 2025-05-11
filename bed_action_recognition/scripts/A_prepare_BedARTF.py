import os
import json
from glob import glob

# ---------- CONFIG ----------
MAIN_FOLDER = "/home/quinoa/Desktop/dccv_act_recog_sequences"  # Replace with your actual path
SEQ_LEN = 7                          # Sequence length (must match your dataset assumptions)
OUT_JSON = 'dataset_sequences.json'  # Output file

# ---------- HELPER ----------
def sorted_glob(folder, pattern):
    return sorted(glob(os.path.join(folder, pattern)))

# ---------- BUILD METADATA ----------
def build_sequence_metadata(main_dir, seq_len):
    sequence_list = []

    for action_class in sorted(os.listdir(main_dir)):
        action_dir = os.path.join(main_dir, action_class)
        if not os.path.isdir(action_dir):
            continue

        for seq_name in sorted(os.listdir(action_dir)):
            seq_dir = os.path.join(action_dir, seq_name)
            rgb_dir = os.path.join(seq_dir, 'RGB')
            depth_dir = os.path.join(seq_dir, 'Depth')

            if not (os.path.isdir(rgb_dir) and os.path.isdir(depth_dir)):
                continue

            rgb_imgs = sorted_glob(rgb_dir, '*.jpg')
            depth_imgs = sorted_glob(depth_dir, '*.png')

            if len(rgb_imgs) < seq_len or len(depth_imgs) < seq_len:
                continue  # Skip incomplete sequences

            entry = {}
            for i in range(seq_len):
                entry["RGB_{:02d}".format(i)] = os.path.relpath(rgb_imgs[i], start=main_dir)
                entry["Depth_{:02d}".format(i)] = os.path.relpath(depth_imgs[i], start=main_dir)
            entry["Action_class"] = action_class
            entry["Sequence_path"] = os.path.relpath(seq_dir, start=main_dir)

            sequence_list.append(entry)

    return sequence_list

if __name__=="__main__":
    JSON_FNAME = "data_bed_action.ignore.json"
    json_dataset = build_sequence_metadata(MAIN_FOLDER,SEQ_LEN) 
    with open(f"bed_action_recognition/data/{JSON_FNAME}","w") as file:
        json.dump(json_dataset,file)
    print(len(json_dataset))
    print(json_dataset[0])
