import tensorflow as tf
from keras import Input as InputTensor
from bed_action_recognition.models.bedar_model import BedARModel

class BedARHTModelHandler():
    def __init__(self,seq_len: int = None,
                 pose_model_path: str = None,
                 input_nchannels: int = 4,
                 num_actions: int = None,
                 batch_size: int = 25,
                 pose_emb_size: int = None,
                 pose_njoints: int = None) -> None:
        self.executed = False
        self.seq_len = seq_len 
        self.pose_model_path = pose_model_path #"data/model_t/myModel_SLP_WS_BL_WithAtt_Depth4C_FT2.keras"
        self.input_nchanels = input_nchannels
        self.batch_size = batch_size
        self.n_actions = num_actions
        self.pose_emb_size = pose_emb_size
        self.pose_njoints = pose_njoints
    def _build_input(self, *args, **kwargs) -> tf.Tensor:
        ## The Height and Width are hardcoded to 256
        # TODO: include the input size in the config file
        height, width = 256,256
        frames = self.seq_len
        channels = self.input_nchanels
        self._input = InputTensor(shape=(frames,height, width, channels), batch_size=self.batch_size, name="InputBedAR")
        return self._input

    def _build_model_as_model(self, *args, **kwargs) -> BedARModel:
        armodel = BedARModel(pose_model_path=self.pose_model_path,
                              n_actions=self.n_actions,
                              channel_num=self.input_nchanels,
                              pose_emb_size=self.pose_emb_size,
                              sequence_len=self.seq_len,
                              pose_njoints=self.pose_njoints, 
                              trainable=True)
        self._model = armodel.wrap_model()
        self._layered_model = {}
        return self._model

    def generate_graph(self, *args, **kwargs) -> None:
        # Get Input Tensor
        input_tensor = self._build_input(*args, **kwargs)
        # Build Model Graph
        print(f"Generating new model from scratch")
        #model = self._build_model_as_model()
        model = self._build_model_as_model(*args, **kwargs)
        model.summary()
        self._model = model
        # Link Input Shape to Model
        self._output = model(inputs=input_tensor, *args, **kwargs)
    def run(self, *args, **kwargs) -> None:
        self.generate_graph(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if not self.executed:
            self.run(*args, **kwargs)
            self.executed = True
        else:
            print(
                f"[WARNING] This {self.__class__.__name__} has already been executed. Use self.reset"
            )
        return self 
    