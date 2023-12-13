from typing import Dict
from typing import List
from typing import Union
from typing import Literal
from typing import Iterable
from typing import Optional

from pydantic import Field
from pydantic import BaseModel


class HTFMetadata(BaseModel):
    available_images: List[str] = Field(default=list)
    label_type: Optional[Union[Literal["json"], Literal["csv"]]] = Field(default="json")
    label_headers: Optional[List[str]] = Field(default=list)
    label_mapper: Optional[Dict[str, int]] = Field(default=dict)
    train_images: Optional[Iterable[str]] = Field(default=list)
    test_images: Optional[Iterable[str]] = Field(default=list)
    validation_images: Optional[Iterable[str]] = Field(default=list)
    joint_columns: Optional[List[str]] = Field(default=list)

    class Config:
        extra = "allow"
