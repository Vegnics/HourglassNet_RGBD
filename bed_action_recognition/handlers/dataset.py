## Read the JSON file containing the paths to the data points.
import numpy as np
import pandas as pd
import tensorflow as tf
import sys,os
sys.path.insert(1,os.getcwd())
from typing import List,Union,Literal,Set,Optional,Tuple,Iterable,Dict
from bed_action_recognition.handlers.engines import BedARTFPandasEngine
from hourglass_tensorflow.utils.sets import split_train_test
from hourglass_tensorflow.handlers._transformation import tf_train_map_build_sequence_Depth
from hourglass_tensorflow.handlers._transformation import tf_check_samples
import random

BedARDataTypes = Union[np.ndarray, pd.DataFrame]
ImageSetsType = Tuple[Optional[Set[str]], Optional[Set[str]], Optional[Set[str]]]
DataModes = Union[Literal["RGB"],Literal["Depth"],Literal["RGBD"]]


class BedARTF_DatasetHandler():
    def __init__(self,
                 data: BedARDataTypes,
                 meta: Dict,
                 data_mode: DataModes):
        # Load dataset config
        self.data = data
        self.context_length: int = 7
        self.n_actions: int = 9
        self.engine = BedARTFPandasEngine()
        self.meta: Dict = meta
        self.task_mode = "train"
        self.data_mode = data_mode
        self.sets: Dict = {"train":True,
                     "validation":True,
                     "test":True,
                     "ratio_train":0.5,
                     "ratio_validation":0.19,
                     "ratio_test":0.01}
        
        self._train_set = None
        self._validation_set = None
        self._test_set = None
        self._executed = False
    
    @property
    def executed(self)->bool:
        return self._executed

    @property
    def has_train(self) -> bool:
        return self.sets["train"]

    @property
    def has_test(self) -> bool:
        return self.sets["test"]

    @property
    def has_validation(self) -> bool:
        return self.sets["validation"]

    @property
    def ratio_train(self) -> float:
        return self.sets["ratio_train"] if self.has_train else 0.0

    @property
    def ratio_test(self) -> float:
        return self.sets["ratio_test"] if self.has_test else 0.0

    @property
    def ratio_validation(self) -> float:
        return self.sets["ratio_validation"] if self.has_validation else 0.0
    
    def _split_sets(self) -> None:
        train, test, validation = self._split_by_ratio()
        self._train_set = train
        self._test_set = test
        self._validation_set = validation
        self.splitted = True

    def _split_by_ratio(self) -> Tuple[BedARDataTypes, BedARDataTypes, BedARDataTypes]:
        # Get set of unique images
        seq_paths = self.engine.get_sequences(data=self.data, column="Sequence_path")
        seq_train, seq_test, seq_validation = self._generate_seq_sets(seq_paths)
        # Select Subsets within the main data collection
        test = self.engine.select_subset_from_sequences(
            data=self.data, sequence_set=seq_test, column="Sequence_path"
        )
        train = self.engine.select_subset_from_sequences(
            data=self.data, sequence_set=seq_train, column="Sequence_path"
        )
        validation = self.engine.select_subset_from_sequences(
            data=self.data, sequence_set=seq_validation, column="Sequence_path"
        )
        return train, test, validation

    def _generate_seq_sets(self, seq_paths: Set[str]) -> ImageSetsType:
        # Images: set of paths to the images.
        # Generate Image Sets: Train, Test, Validation.
        train = set()
        test = set()
        validation = set()
        if self.has_train:
            # Has training samples
            if self.has_test & self.has_validation:
                print("TEST AND VALIDATION")
                # + Validation and Test
                train, test = split_train_test(
                    seq_paths, self.ratio_train + self.ratio_validation,self.ratio_test
                )
                train, validation = split_train_test(
                    train, self.ratio_train / (self.ratio_train + self.ratio_validation)
                )
            elif self.has_validation:
                train, validation = split_train_test(seq_paths, self.ratio_train)
            elif self.has_test:
                train, test = split_train_test(seq_paths, self.ratio_train)
            else:
                train = seq_paths
        else:
            if self.has_test & self.has_validation:
                test, validation = split_train_test(seq_paths, self.ratio_test)
            elif self.has_test:
                test = seq_paths
            else:
                validation = seq_paths
        return train, test, validation
    
    def prepare_dataset(self, *args, **kwargs) -> None:
        self._split_sets()
        print(f"Samples train: {len(self._train_set)}\n Samples test: {len(self._test_set)}\n Samples validation{len(self._validation_set)}")

    def _extract_columns_from_data(
        self, data: BedARDataTypes
    ) -> Union[Tuple[Iterable, Iterable],Tuple[Iterable,Iterable,Iterable]]:
        # Extract the columns containing the joints' locations
        depth_cols = [elem for elem in self.meta["label_headers"] if elem.startswith("Depth")]
        rgb_cols = [elem for elem in self.meta["label_headers"] if elem.startswith("RGB")]
        
        action_ids = self.engine.to_list(
            self.engine.get_columns(data=data, columns=["Action_class"])
        )
        # Extract the columns containing the filenames of the RGB images
        rgb_fnames = self.engine.to_list(
            self.engine.get_columns(data=data, columns=rgb_cols)
        )

        if self.data_mode == "RGB":
            print("LEN_FILENAMES: ",len(rgb_fnames),"LEN_SEQS :",len(action_ids))
            return rgb_fnames, action_ids
        
        elif self.data_mode == "RGBD":
            depth_fnames = self.engine.to_list(
                self.engine.get_columns(data=data, columns=depth_cols)
            )
            _extracted_data = list(zip(rgb_fnames,depth_fnames,action_ids))
            random.shuffle(_extracted_data)
            rgb_fnames,depth_fnames,action_ids = zip(*_extracted_data)
            rgb_fnames = list(rgb_fnames)
            depth_fnames = list(depth_fnames)
            action_ids = list(action_ids)
            return rgb_fnames,depth_fnames,action_ids
        
        elif self.data_mode == "Depth":
            depth_fnames = self.engine.to_list(self.engine.get_columns(data=data, 
                                                                       columns=depth_cols))
            _extracted_data = list(zip(depth_fnames,action_ids))
            random.shuffle(_extracted_data)
            depth_fnames,action_ids = zip(*_extracted_data)
            depth_fnames = list(depth_fnames)
            action_ids = list(action_ids)
            return depth_fnames,action_ids
        else:
            raise Exception(f"The data_mode {self.data_mode } is not valid.")

    def _create_dataset_train_eval(self,data):
        """
        Load images, and apply transformations to the data and annotations.
        """
        #raw = tf.data.Dataset.from_tensor_slices(self._extract_columns_from_data(data=data[0:10])) #fname, coordinates
        raw = tf.data.Dataset.from_tensor_slices(self._extract_columns_from_data(data=data))
        #TODO: raw.map for RGB and RGBD modalities
        if self.data_mode == "RGB":
            raw = raw.map(lambda rgb,action_id:(rgb,tf.one_hot(action_id,self.n_actions)),num_parallel_calls=10)
        elif self.data_mode == "Depth":
            raw = raw.map(lambda depth,action_id:(depth,tf.one_hot(action_id,self.n_actions)),num_parallel_calls=10)
            raw = raw.map(lambda d_map,action_encoded:tf_train_map_build_sequence_Depth(d_map,action_encoded))
        elif self.data_mode == "RGBD":
            raw = raw.map(lambda rgb,depth,action_id:(rgb,depth,tf.one_hot(action_id,self.n_actions)),num_parallel_calls=10)
        else:
            raise Exception("Invalid data mode")
        print("RAW1:::",raw)

        return raw
    def _create_dataset_test(self,data):
        """
        Load images, and apply transformations to the data and annotations.
        """
        #raw = tf.data.Dataset.from_tensor_slices(self._extract_columns_from_data(data=data[0:10])) #fname, coordinates
        raw = tf.data.Dataset.from_tensor_slices(self._extract_columns_from_data(data=data))
        return raw
    
    def generate_datasets(self, *args, **kwargs) -> None:
        if self.task_mode == "train":
            self._train_dataset = self._create_dataset_train_eval(self._train_set)
            #self._train_dataset = self._train_dataset.shuffle(1800,reshuffle_each_iteration=True)
            #print("TRAIN DATASET: ",self._train_dataset)
            self._test_dataset = self._create_dataset_test(self._test_set)
            #self._test_dataset.shuffle(self._test_dataset.cardinality(),reshuffle_each_iteration=False)
            self._validation_dataset = self._create_dataset_train_eval(self._validation_set)
            #self._validation_dataset.shuffle(self._validation_dataset.cardinality(),reshuffle_each_iteration=False)
        elif self.task_mode == "test":
            self._test_dataset = self._create_dataset_test(self._test_set)
    def run(self, *args, **kwargs) -> None:
        self.prepare_dataset(*args, **kwargs)
        self.generate_datasets(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        if not self.executed:
            self.run(*args, **kwargs)
            self._executed = True
        else:
            print(
                f"[WARNING] This {self.__class__.__name__} has already been executed. Use self.reset"
            )
        return self 