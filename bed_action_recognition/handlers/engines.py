import pandas as pd
from typing import Set,List,Union

class BedARTFPandasEngine():
    def __init__(
        self, *args, **kwargs
    ) -> None:
        self.kwargs = kwargs

    def get_sequences(self, data: pd.DataFrame, column: str) -> Set[str]:
        sequences: Set[str] = set(data[column].tolist())
        return sequences

    def filter_data(
        self, data: pd.DataFrame, column: str, set_name: str
    ) -> pd.DataFrame:
        return data.query(f"{column} == '{set_name}'")

    def select_subset_from_sequences(
        self, data: pd.DataFrame, sequence_set: Set[str], column: str
    ) -> pd.DataFrame:
        return data[data[column].isin(sequence_set)]

    def get_columns(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return data[columns] #if len(columnsn charge of receiving the 
    
    @staticmethod
    def to_list(data: pd.DataFrame) -> List:
        return data.values.tolist()