# Parsers

## MPII parser

### parse_mpii()
*hourglass_tensorflow/utils/parsers/mpii.py*

Read a MATLAB structure and convert it into a MPII Object (`Dict[str,list]`, `MPIIDataset`, `List[MPIIDatapoint]`). The resulting MPII Object still has the MPII format and must be transformed into a `List` of `HTFPersonDatapoint`.  

## HTF parser
*hourglass_tensorflow/utils/parsers/htf.py*

### from_train_mpii_to_htf_data()

Converts the structured MPII object (containing the training data) into a `list` of `HTFPersonDatapoint`. One `HTFPersonDataPoint` contains the annotations for a single person, and several `HTFPersonDataPoint` may refer to the same image, but with different bounding boxes and landmark annotations. Resulting `list` can be saved as a *.json* file. 

The generated *.json* file contains the annotations from the dataset (in this case MPII) with a structure tailored to be used by the Hourglass model. 

In the following code the `HTFPersonDataPoint` instances are converted into `Dict` with a simplified structure. This "simple" structure can be explicitly used to train and test an Hourglass Network model regardless of the structure or annotation format provided in the initial dataset (in this case MPII). 

```python
"""
Taken from: scripts/B_prepare_htf_data.py
"""
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

```
At the end of the code above, the new simplified dataset is saved as a *.json* file. The path to this *.json* file should be included in the **Training configuration File** 

```yaml
# Example of the Training configuration File, using a YAML format.
mode: train
version: "1.0.0"
data:
...
...
output:
    source: PATH_TO_THE_HG_DATASET_JSON_FILE
...
...
```
>Thus, the final dataset (image filenames/paths + annotations) must be created using two parsers: 
 >- First parser:  Converts the RAW data annotations (usually these annotations are language agnostic) into a structure that can be easily accessed in Python. 
 >- Second parser: Converts the Python-tailored data into `HTFPersonDataPoint` instances. Then these `HTFPersonDataPoint` objects are converted into a `List` of `Dict` with the HG simple data structure and stored.    


