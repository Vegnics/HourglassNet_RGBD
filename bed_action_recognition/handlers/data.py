import itertools
import os
#from hourglass_tensorflow.utils import ObjectLogger
import pandas as pd
from typing import List

MAIN_FOLDER = "/home/quinoa/Desktop/dccv_act_recog_sequences"
JSON_FNAME = "data_bed_action.ignore.json"

class BedARHTDataHandler():
    ## Generate a pd.dataframe from the datapoints' JSON 
    def __init__(self,
                 main_folder: str,
                 db_json: str,
                 sequence_length: int):
        self.input_cfg = {"source":main_folder,"seq_len":sequence_length}
        self.output_cfg = {"source":db_json,"prefix_columns":["Action_class","Sequence_path"],"source_prefixed":False}
        self.meta = {"label_headers":None,"label_type":None}
        self._executed = False
    @property
    def executed(self)->bool:
        return self._executed
    
    def _valid_labels_header(self, df: pd.DataFrame, _error: bool = False) -> bool:
        # Check if columns names are valid
        seq_headers = self._get_seq_columns()
        headers = [*self.output_cfg["prefix_columns"], *seq_headers]
        if set(headers).difference(set(list(df.columns))):
            if _error:
                raise Exception(
                    f"Columns' name does not match configuration\n\tEXPECTED:\n\t{headers}\n\tRECEIVED:\n\t{list(df.columns)}\n\tMISSING COLUMNS:\n\t{set(headers).difference(set(list(df.columns)))}"
                )
            return False
        # If everything is good we store the expected headers in _metadata
        self.meta["label_headers"] = headers
        return True

    def _load_labels(self) -> pd.DataFrame:
        print(f"[INFO] Reading labels from {self.output_cfg['source']}")
        ## Check if the file extension is in [.json, .csv]
        if self.output_cfg["source"].endswith(".json"):
            self.meta["label_type"] = "json"
            labels = pd.read_json(self.output_cfg["source"], orient="records")
        elif self.output_cfg["source"].endswith(".csv"):
            self.meta["label_type"] = "csv"
            labels = pd.read_csv(self.output_cfg["source"])
        else:
            raise Exception(
                f"{self.output_cfg['source']} should be of type .json or .csv"
            )
        if not isinstance(labels, pd.DataFrame):
            raise Exception(
                f"{self.output_cfg['source']} not parsable as pandas.DataFrame"
            )
        return labels

    def _prefix_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        # Now we also prefix the image column with the image folder
        # in case the source_prefix attribute is set to false
        folder_prefix = self.input_cfg["source"]
        source_columns = self._get_seq_columns()
        source_columns += ["Sequence_path"]
        #Prefix RGB and Depth
        for scol in source_columns:
            df = df.assign(
                **{
                    scol: df[scol].apply(
                        lambda x: os.path.join(folder_prefix, x)
                    )
                }
            )
        return df

    def _read_labels(self, _error: bool = False) -> bool:
        # Check if data.output.source exists ?
        if not os.path.exists(self.output_cfg["source"]):
            raise Exception(f"Unable to find {self.output_cfg['source']}")
        # Read Data
        labels = self._load_labels()

        ##### CHECK THE LABELS ######
        # Validate expected labels columns
        if not self._valid_labels_header(labels, _error=_error):
            #self.error("Labels are not matching")
            print("[ERROR] Labels are not matching")
            return False
        
        self._labels_df: pd.DataFrame = labels[self.meta["label_headers"]]
    
        if not self.output_cfg["source_prefixed"]:
            self._labels_df = self._prefix_sequences(self._labels_df)
        return True

    def _get_seq_columns(self) -> List[str]:
        modalities = ["RGB","Depth"]
        frame_idxs = range(self.input_cfg["seq_len"])
        prod = itertools.product(modalities,frame_idxs)
        return ["{}_{:02d}".format(mod,idx)for mod,idx in prod]

    def prepare_output(self, _error: bool = True) -> None:
        # Read the label file
        self._read_labels(_error=_error)

    def run(self, *args, **kwargs) -> None:
        self.prepare_output(*args, **kwargs)
        
    def get_data(self) -> pd.DataFrame:
        return self._labels_df
    
    def __call__(self, *args, **kwargs):
        if not self.executed:
            self.run(*args, **kwargs)
            self._executed = True
        else:
            print(
                f"[WARNING] This {self.__class__.__name__} has already been executed. Use self.reset"
            )
        return self
    
if __name__ == "__main__":
    _data_handler = BedARHTDataHandler(MAIN_FOLDER,f"bed_action_recognition/data/{JSON_FNAME}")
    _data_handler()
    print(_data_handler.get_data()[0:10])
    print(_data_handler.meta)