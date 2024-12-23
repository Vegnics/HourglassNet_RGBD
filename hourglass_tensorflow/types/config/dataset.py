from typing import Union
from typing import Literal
from typing import Optional

from pydantic import Field
from pydantic import BaseModel

from hourglass_tensorflow.types.config.fields import HTFConfigField
from hourglass_tensorflow.types.config.fields import HTFObjectReference

NormalizationModeType = Union[
    Literal["Idem"],
    Literal["ByMax"],
    Literal["L2"],
    Literal["Normal"],
    Literal["FromZero"],
    Literal["AroundZero"],
]


class HTFDatasetSets(HTFConfigField):
    """
    Configuration for the Train, Test, Validation sets.
    """
    split_by_column: bool = False
    column_split: str = "set"
    value_test: str = "TEST"
    value_train: str = "TRAIN"
    value_validation: str = "VALIDATION"
    test: bool = False
    train: bool = True
    validation: bool = True
    ratio_test: float = 0.0
    ratio_train: float = 0.8
    ratio_validation: float = 0.2


class HTFDatasetBBox(HTFConfigField):
    activate: bool = True
    factor: float = 1.0


class HTFDatasetHeatmap(HTFConfigField):
    size: int = 64
    stacks: int = 3
    channels: int = 16
    stddev: float = 16
    stddev_factor: float = 1.5

class HTFDatasetHipIndexes(HTFConfigField):
    Lhip: int = 3
    Rhip:int = 2

class HTFDatasetMetadata(BaseModel):
    class Config:
        extra = "allow"


class HTFDatasetConfig(HTFConfigField):
    """
    Configuration of the dataset: Image size, Heatmap configuration, Sets configuration, 
    bounding box configuration.
    """
    object: Optional[HTFObjectReference] = Field(
        default=HTFObjectReference(
            source="hourglass_tensorflow.handlers.dataset.HTFDatasetHandler"
        )
    )
    image_size: int = 256
    column_image: str = "image"
    column_depth_image: Optional[str] = Field(default="depth") # ADDED FOR THE RGBD VERSION
    data_mode: Optional[str] = Field(default="RGB") # ADDED FOR THE RGBD VERSION
    task_mode: Optional[str] = Field(default="train") # ADDED FOR MODEL EVALUATION
    heatmap: Optional[HTFDatasetHeatmap] = Field(default=HTFDatasetHeatmap)
    hip_idxs: Optional[HTFDatasetHipIndexes] = Field(default=HTFDatasetHipIndexes)
    sets: Optional[HTFDatasetSets] = Field(default=HTFDatasetSets)
    bbox: Optional[HTFDatasetBBox] = Field(default=HTFDatasetBBox)
    normalization: Optional[NormalizationModeType] = None
