import sys,os
sys.path.insert(1,os.getcwd())
from bed_action_recognition.handlers.data import BedARHTDataHandler 
from bed_action_recognition.handlers.dataset import BedARTF_DatasetHandler
from bed_action_recognition.handlers.train import BedARTF_TrainHandler
from bed_action_recognition.handlers.model import BedARHTModelHandler
from matplotlib import pyplot as plt



class BedARHTManager():
    def __init__(self,
                 db_folder: str,
                 db_json: str,
                 train_epochs: int,
                 train_batch_size: int,
                 sequence_length: int = None,
                 pose_model_path: str = None,
                 action_num: int = None,
                 pose_njoints: int = None,
                 pose_emb_size: int = None):
        self.notthing = None
        #self.data_handler = BedARHTDataHandler(MAIN_FOLDER,f"bed_action_recognition/data/{JSON_FNAME}")
        self.data_handler = BedARHTDataHandler(db_folder,
                                               db_json,
                                               sequence_length=sequence_length)
        self.train_handler = BedARTF_TrainHandler(epochs=train_epochs,
                                                  batch_size=train_batch_size)
        self.dataset_handler = None
        self.model_handler = BedARHTModelHandler(seq_len=sequence_length,
                                                 pose_model_path=pose_model_path,
                                                 num_actions=action_num,
                                                 batch_size=train_batch_size,
                                                 pose_njoints= pose_njoints,
                                                 pose_emb_size=pose_emb_size)
    def train(self):
        _data = self.data_handler().get_data()
        #print(self.data_handler.get_data()[0:10])
        #print(self.data_handler.meta)
        self.dataset_handler = BedARTF_DatasetHandler(_data,self.data_handler.meta,"Depth")
        self.dataset_handler()
        """
        for k,(seq,action) in enumerate(self.dataset_handler._train_dataset):
            for i in range(7):
                plt.imshow(seq[i,:,:,0],cmap="jet")
                plt.show()
            if k>100:
                break
        """
        _train_dataset = self.dataset_handler._train_dataset
        #print("TRAIN_DATASET:::",_train_dataset)
        _validation_dataset = self.dataset_handler._validation_dataset
        _test_dataset = self.dataset_handler._test_dataset
        self.model_handler()
        model = self.model_handler._model
        self.train_handler.compile(model=model)
        self.train_handler.fit(model,train_dataset=_train_dataset,
                               test_dataset=_test_dataset,
                               validation_dataset=_validation_dataset)
        

    
if __name__ == "__main__":
    manager = BedARHTManager()
    manager.train()


