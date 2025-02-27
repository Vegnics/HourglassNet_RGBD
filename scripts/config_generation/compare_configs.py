import yaml

cfg_name = "config/train_rgbd.default.yaml"
outcfg_name = "config/train_rgbdGEN.default.yaml"

with open(cfg_name, "r") as f:
    conf = yaml.safe_load(f)

with open(outcfg_name, "r") as f:
    conf2 = yaml.safe_load(f)

print(conf,"\n\n")
print(conf2)



import json

dict1 = {...}  # First dictionary (paste here)
dict2 = {...}  # Second dictionary (paste here)

# Compare JSON dumps for quick string-based diff
if json.dumps(conf, sort_keys=True) == json.dumps(conf2, sort_keys=True):
    print("✅ The dictionaries are identical!")
else:
    print("❌ The dictionaries have differences!")

# Deeper comparison
def find_diff(d1, d2, path=""):
    if isinstance(d1, dict) and isinstance(d2, dict):
        for key in d1.keys() | d2.keys():  # Union of keys
            new_path = f"{path}.{key}" if path else key
            if key not in d2:
                print(f"❌ Missing in dict2: {new_path}")
            elif key not in d1:
                print(f"❌ Missing in dict1: {new_path}")
            else:
                find_diff(d1[key], d2[key], new_path)
    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print(f"❌ Different list lengths at {path}: {len(d1)} vs {len(d2)}")
        for i, (item1, item2) in enumerate(zip(d1, d2)):
            find_diff(item1, item2, f"{path}[{i}]")
    elif d1 != d2:
        print(f"❌ Mismatch at {path}: {d1} vs {d2}")

find_diff(conf, conf2)
