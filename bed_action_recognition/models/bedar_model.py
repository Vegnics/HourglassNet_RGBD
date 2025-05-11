from bed_action_recognition.layers.heatmap_seq_ar_layer import ActionRecognitionLayer
from hourglass_tensorflow.models import HourglassModel
from hourglass_tensorflow.utils.loaders.model_loader import load_wrapped_model
from keras.models import Model
from keras.layers import Layer
from keras import Input as InputTensor
import tensorflow as tf
from keras.utils import register_keras_serializable

@register_keras_serializable(package="cBedARModel")
class StackHeatmaps(Layer):
    def call(self, inputs, **kwargs):
        x = tf.stack(inputs, axis=-1) 
        return tf.transpose(x, perm=[0,4,1,2,3])

@register_keras_serializable(package="cBedARModel")
class BedARModel(Model):
    def __init__(
        self,
        input_size: int = 256,
        n_actions: int = 9,
        channel_num: int = 4,
        sequence_len: int = 7,
        pose_model_path: str = None,
        pose_njoints: int = None,
        name: str = "BedARSample",
        trainable: bool = True,
        pose_emb_size: int = None,
        *args,
        **kwargs,
    )-> None:
        super().__init__(name=name,trainable=trainable)
        # Init
        self.input_size = input_size
        self.n_actions = n_actions
        self.pose_model_path = pose_model_path
        self.channelnum = channel_num
        self.pose_model = None
        self.seq_len = sequence_len
        self.pose_njoints = pose_njoints
        self.pose_emb_size = pose_emb_size
     
        # Layers
        self.act_recog_layer = ActionRecognitionLayer(name="act_recog_layer",
                                                      context_T= self.seq_len,
                                                      NJoints=self.pose_njoints,
                                                      NActions=self.n_actions,
                                                      emb_len=self.pose_emb_size,
                                                      trainable=self.trainable)
        self.stack_layer = StackHeatmaps(name="stack_hms_layer")
        self.pose_model = load_wrapped_model(fpath=self.pose_model_path,compile=False)
        self.pose_model.trainable = False

    def _yes_no_str(self,val: bool):
        return "Yes" if val else "No"
    
    def wrap_model(self):
        inputs = InputTensor(shape=(self.seq_len,self.input_size, self.input_size, self.channelnum))
        outputs = self.call(inputs)
        wrapped = Model(inputs, outputs,name="FunctionalBedARModel")
        wrapped.core = self
        original_get_config = wrapped.get_config
        def merged_get_config():
            config = original_get_config()
            config["core_config"] = self.get_config()
            return config

        wrapped.get_config = merged_get_config
        return wrapped

    def build(self,input_shape=None):
        if input_shape is None:
            T = self.seq_len
            N = self.input_size
            C = self.channelnum
            _input_shape = (None,T,N,N,C)
            super().build(_input_shape)
        else:
            super().build(input_shape)
            _input_shape = input_shape
        self.built = True
        # You can print the input shape to verify it
        print("---------Model configuration summary------------")
        print(f'Building model with input shape: {_input_shape}')
    
    def call(self, inputs: tf.Tensor, training=True): #Inputs are (B,7,256,256,4)
        out_hms = []
        for k in range(self.seq_len):
            heatmaps = self.pose_model(inputs[:,k,:,:,:], training=False)
            out_hms.append(heatmaps[:,-1,:,:,0:self.pose_njoints])
        out_hms = self.stack_layer(out_hms)
        out_actions = self.act_recog_layer(out_hms,training=training)
        return out_actions
    
    def get_config(self):
        return {
            "input_size": self.input_size,
            "n_actions": self.n_actions,
            "channel_num": self.channelnum,
            "pose_model_path": self.pose_model_path,
            "name": self.name,
            "trainable": self.trainable,
            "sequence_len" :self.seq_len,
            "pose_njoints": self.pose_njoints,
            "pose_emb_size": self.pose_emb_size
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    