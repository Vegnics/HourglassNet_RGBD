# HTF Datatypes
It uses the BaseModel from the Pydantic library. 

## HTFPoint(BaseModel)
*hourglass_tensorflow/types/htf_data_types.py*

- Properties:
  - `int` x
  - `int` y

## HTFPersonBBox(BaseModel)
*hourglass_tensorflow/types/htf_data_types.py*

- Properties:
  - `HTFPoint` top_left
  - `HTFPoint` bottom_right

## HTFPersonJoint(HTFPoint)
*hourglass_tensorflow/types/htf_data_types.py*

- Properties:
  - `int` x
  - `int` yhourglass_tensorflow/types/htf_data_types.pyble

## HTFPersonDatapoint(BaseModel)
*hourglass_tensorflow/types/htf_data_types.py*

- Properties:
  - `int` is_train
  - `int` image_id
  - `int` person_id
  - `str` source_image
  - `HTFPersonBBox` bbox
  - `List[HTFPersonJoint]` or `Dict[int, HTFPersonJoint]]` joints
- Methods:
  - **convert_joint**(to): Convert the joints property into a `list` or `dict`.

## class HTFConfigMode(enum.Enum):
*hourglass_tensorflow/types/config/__init\__.py*

    TEST = "test"
    TRAIN = "train"
    INFERENCE = "inference"
    SERVER = "server"

## HTFConfigField(BaseModel)
*hourglass_tensorflow/types/config/fields.py*

- Properties:
  - ` List[bool]` VALIDITY_CONDITIONS
  - `bool` is_valid

## HTFDataConfig(HTFConfigField)
*hourglass_tensorflow/types/config/data.py*

- Properties:
  - `HTFDataInput` input : Contains information about the source folder containing the images, and additional info about the images.
  - `HTFDataOutput` output: Contains information about the source file containing the annotations, and the joints annotation format (naming convention, number of joints, joint names).
  - `HTFObjectReference` object: Reference to a data handler.

## HTFDatasetConfig(HTFConfigField)
*hourglass_tensorflow/types/config/dataset.py*

- Properties:
  - `int` image_size
  - `str` column_image: Name of the column containing the image filenames.
  - `HTFDatasetHeatmap` heatmap: Parameters related to the heatmap generation.
  - `HTFDatasetSets` sets: Parameters related to the way test, train, validation sets are generated (ratios).
  - `HTFDatasetBBox` bbox: Parameters related to the bounding box. 
  - `str` normalization: *'Bymax'*, *'L2'*, *'Normal'*, *'FromZero'*, *'AroundZero'*.

## HTFModelConfig(HTFConfigField)
*hourglass_tensorflow/types/config/model.py*

- Properties:
  - `bool` build_as_model
  - `str` data_format: NHWC (*None, Height, Width, Channels*).
  - `HTFModelParams` params: Contains the parameters defining the stacked Hourglass network architecture :
     - input_size
     - stages
     - stage_filters: filters per stage
     - output_channels: Number of heatmaps at the output.
     - downsamplings_per_stage 
     - intermediate_supervision.

## HTFTrainConfig(HTFConfigField)
*hourglass_tensorflow/types/config/train.py*

- Properties: 
  - `int` epochs
  - `int` epoch_size
  - `int` batch_size: `[Optional]`
  - `str` or `HTFObjectReference[Loss]` loss: loss function can be specified by name or by its `HTFObjectReference`.
  - `str` or `HTFObjectReference[optimizer]` optimizer: optimizer can be specified by name or by its `HTFObjectReference`.
  - `List` callbacks: a list containing the `HTFObjectReference` instances to `Callback` instances.
  - `List` metrics: a list containing the `HTFObjectReference` instances to `Metric` instances.
  - `HTFObjectReference` object: The reference to `HTFTrainHandler`. 

## HTFConfig(HTFConfigField)
*hourglass_tensorflow/types/config/__init\__.py*

- Properties:
  - `HTFConfigMode` mode
  - `int` or `str` version
  - `HTFDataConfig` data
  - `HTFDatasetConfig` dataset
  - `HTFModelConfig` model
  - `HTFTrainConfig` train
