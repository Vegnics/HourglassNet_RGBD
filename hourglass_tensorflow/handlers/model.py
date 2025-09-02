from abc import abstractmethod

from numpy import True_
import tensorflow as tf
import keras.layers
import keras.models
from keras import Input as InputTensor
import inspect
from keras.layers import Layer
from hourglass_tensorflow.utils.loaders.weight_loader import print_layers_recursive


from hourglass_tensorflow.utils import BadConfigurationError
from hourglass_tensorflow.models.hourglass_lora import HourglassModelLora,build_hourglassModelLora,model_as_layers_lora, WrappedModel
#from hourglass_tensorflow.models.hourglass_lora import model_as_layers
#from hourglass_tensorflow.models.hourglass import HourglassModel,build_hourglassModel,model_as_layers

from hourglass_tensorflow.types.config import HTFModelConfig
from hourglass_tensorflow.types.config import HTFModelParams
from hourglass_tensorflow.types.config import HTFModelHandlerReturnObject
from hourglass_tensorflow.handlers.meta import _HTFHandler

from hourglass_tensorflow.metrics.correct_keypoints import PercentageOfCorrectKeypoints
from hourglass_tensorflow.metrics.distance import OverallMeanDistance
from hourglass_tensorflow.losses.mae_custom import MAE_custom
from hourglass_tensorflow.metrics import SoftargmaxMeanDist
from hourglass_tensorflow.utils.loaders.model_loader2 import load_wrapped_model,load_basemodel_weights
from hourglass_tensorflow.layers.downsampling_lora import DownSamplingLayerLora
from hourglass_tensorflow.layers.hourglass_lora import HourglassLayerLora

# region Abstract Class


class _HTFModelHandler(_HTFHandler):
    def __init__(
        self,
        config: HTFModelConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config=config, *args, **kwargs)
        self._input: keras.layers.Layer = None
        self._output: keras.layers.Layer = None
        self._model: keras.models.Model = None

        # ADDED JUST FOR TESTING CONSISTENCY AFTER LOADING A MODEL
        #self._dummy_model: keras.models.Model = None


    @property
    def config(self) -> HTFModelConfig:
        return self._config

    @property
    def params(self) -> HTFModelParams:
        return self.config.params

    def get(self) -> HTFModelHandlerReturnObject:
        if self._executed:
            return {
                "inputs": self._input,
                "outputs": self._output,
                "model": self._model,
            }
        else:
            self.warning(
                "The ModelHandler has not been called to generate proper return value"
            )
            return {}

    @abstractmethod
    def generate_graph(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        self.generate_graph(*args, **kwargs)


# enregion

# region Handler


class HTFModelHandler(_HTFModelHandler):
    def init_handler(self, *args, **kwargs) -> None:
        pass

    def get(self) -> HTFModelHandlerReturnObject:
        if self._executed:
            return {
                "inputs": self._input,
                "outputs": self._output,
                "model": self._model,
                "layers": self._layered_model,
            }
        else:
            self.warning(
                "The ModelHandler has not been called to generate proper return value"
            )
            return {}

    def _build_input(self, *args, **kwargs) -> tf.Tensor:
        height, width = self.params.input_size, self.params.input_size
        # TODO: Handle other Image Mode than RGB
        channels =  int(self.params.channel_number)
        #channels = 4
        if self.config.data_format == "NHWC":
            self._input = InputTensor(shape=(height, width, channels), batch_size=self.config.batch_size, name="InputHG")
        else:
            raise BadConfigurationError("The only supported data format is NHWC so far")
        return self._input

    def _build_model_as_model(self, *args, **kwargs) -> HourglassModelLora:
        hgmodel = HourglassModelLora(**self.params.model_dump())
        self._model = hgmodel.wrap_model()
        #self._dummy_model = HourglassModel(**self.params.model_dump())
        self._layered_model = {}
        return self._model

    def _build_model_as_layer(self, *args, **kwargs) -> keras.models.Model:
        self._layered_model = model_as_layers_lora(inputs=self._input, **self.params.model_dump())
        self._output = self._layered_model["outputs"]
        self._model = self._layered_model["model"]
        return self._model

    def generate_graph(self, *args, **kwargs) -> None:
        print("PARAMS",self.params)
        # Get Input Tensor
        input_tensor = self._build_input(*args, **kwargs)
        # Build Model Graph
        if self.config.build_as_model:
            if not self.config.load_model:
                print(f"Generating new model from scratch")
                #model = self._build_model_as_model()
                model = self._build_model_as_model(*args, **kwargs)
                #print_layers_recursive(model,path="")
                #print(model.get_config())
                #print(model.stages,model.channels_1J,model.channels_2J)
                #custom_objects = {name: obj for name, obj in inspect.getmembers(hourglass_tensorflow.layers, inspect.isclass) if issubclass(obj, tf.keras.layers.Layer)}
                #print(custom_objects)
            else:
                if not (self.config.loading_style == "Frozen"):
                    model = self._build_model_as_model(*args, **kwargs)
                    model(tf.random.normal((1, 256, 256, self.params.channel_number)))
                    print(f"Loading base model ... {self.config.model_path}")
                    load_basemodel_weights(model,self.config.model_path,compile=False)
                else:
                    model = load_wrapped_model(self.config.model_path,compile=False)

                if self.config.loading_style == "Partial_Train_Joints":
                    print(">>>>>>>>>> [LOADING] PARTIAL TRAINING FOR JOINTS <<<<<<<<<<<<<<")
                    for layer in model.layers:
                        if isinstance(layer,DownSamplingLayerLora):
                            print(f"Freezing {layer.name}")
                            layer.trainable = False
                        elif isinstance(layer,HourglassLayerLora):
                            main_name = layer.name
                            # Freeze the main hourglass
                            for _,val in layer.layer_list.items():
                                for _,v in val.items():
                                    if isinstance(v,Layer):
                                        print(f"Freezing {main_name}/{v.name}")
                                        v.trainable = False
                            print(f"Freezing {main_name}/{layer.residual_brc.name}")
                            layer.residual_brc.trainable = True
                            print(f"Freezing {main_name}/{layer.merge_feats_main.name}")
                            layer.merge_feats_main.trainable = False
                            print(f"Freezing {main_name}/{layer.merge_feats_1j.name}")
                            layer.merge_feats_1j.trainable = False
                            print(f"Freezing {main_name}/{layer.ln_feats1j.name}")
                            layer.bn_feats_1j.trainable = False 
                            layer.residual_2j.activate_lora = False
               
                elif self.config.loading_style == "Partial_Train_Attention":
                    print(">>>>>>>>>> [LOADING] PARTIAL TRAINING FOR ATTENTION <<<<<<<<<<<<<<")
                    for layer in model.layers:
                        if isinstance(layer,DownSamplingLayerLora):
                            print(f"Freezing {layer.name}")
                            layer.trainable = False
                        
                        elif isinstance(layer,HourglassLayerLora):
                            main_name = layer.name
                            # Train only the main hourglass
                            for _,val in layer.layer_list.items():
                                val["up_1"].trainable = True # Freeze the Skip layers
                                #val["up_1"].residual_blocks[0].conv_block.trainable = False
                                print("Freezing {}/{}".format(main_name,val["up_1"].name))
                                val["low_1"].trainable = True # Train S2F
                                #val["low_1"].residual_blocks[0].conv_block.trainable = False # Freeze the ConvBlock
                                print("Freezing {}/{}".format(main_name,val["low_1"].name))
                                val["low_3"].trainable = True # Freeze F2S
                                #val["low_3"].residual_blocks[0].conv_block.trainable = False
                                print("Freezing {}/{}".format(main_name,val["low_3"].name))
                                try:
                                    val["low_2"].trainable = True # Freeze F2S
                                    print("Freezing {}/{}".format(main_name,val["low_2"].name))
                                    #val["low_2"].residual_blocks[0].conv_block.trainable = False
                                except:
                                    pass

                            print(f"Training {main_name}/{layer.residual_brc.name}")
                            layer.residual_brc.trainable = True
                            #layer.residual_brc.residual1.residual_blocks[0].conv_block.trainable = False
                            #layer.residual_brc.lora_path.trainable = True
                            print(f"Freezing {main_name}/{layer.merge_feats_main.name}")
                            layer.merge_feats_main.trainable = False
                            print(f"Freezing {main_name}/{layer.merge_feats_1j.name}")
                            layer.merge_feats_1j.trainable = False #False
                            
                            #print(f"Freezing {main_name}/{layer.ln_inputs.name}")
                            #layer.ln_inputs.trainable = False
                            #print(f"Freezing {main_name}/{layer.ln_main.name}")
                            #layer.ln_main.trainable = False
                            #print(f"Freezing {main_name}/{layer.ln_feats1j.name}")
                            #layer.ln_feats1j.trainable = False 
                            #print(f"Freezing {main_name}/{layer.bn_feats_1j.name}")
                            layer.bn_feats_1j.trainable = False
                            print(f"Freezing {main_name}/{layer.hm1_output.name}")
                            layer.hm1_output.trainable = True #False
                            print(f"Freezing {main_name}/{layer.hm2_output.name}")
                            layer.hm2_output.trainable = False
                            print(f"Freezing {main_name}/{layer.features_hm2.name}")
                            layer.features_hm2.trainable = False
                            print(f"Freezing {main_name}/{layer.residual_2j.name}")
                            layer.residual_2j.trainable = False

                elif self.config.loading_style == "Partial_Train_S2F":
                    print(">>>>>>>>>> [LOADING] PARTIAL TRAINING FOR S2F (Bottom-up) <<<<<<<<<<<<<<")
                    for layer in model.layers:
                        if isinstance(layer,DownSamplingLayer):
                            print(f"Freezing {layer.name}")
                            layer.trainable = False
                        elif isinstance(layer,HourglassLayer):
                            main_name = layer.name
                            # Train only the main hourglass
                            for _,val in layer.layer_list.items():
                                val["up_1"].trainable = False # Freeze the Skip layers
                                print("Freezing {}/{}".format(main_name,val["up_1"].name))
                                val["low_1"].trainable = True # Train S2F
                                val["low_1"].residual_blocks[0].conv_block.trainable = False # Freeze the ConvBlock
                                #val["low_1"].residual_blocks[0].alpha.trainable = False # Freeze the ConvBlock
                                val["low_3"].trainable = False # Freeze F2S
                                print("Freezing {}/{}".format(main_name,val["low_3"].name))
                            print(f"Freezing {main_name}/{layer.residual_brc.name}")
                            layer.residual_brc.trainable = False
                            print(f"Freezing {main_name}/{layer.merge_feats_main.name}")
                            layer.merge_feats_main.trainable = False
                            print(f"Freezing {main_name}/{layer.merge_feats_1j.name}")
                            layer.merge_feats_1j.trainable = False
                            #print(f"Freezing {main_name}/{layer.ln_inputs.name}")
                            #layer.ln_inputs.trainable = False
                            #print(f"Freezing {main_name}/{layer.ln_main.name}")
                            #layer.ln_main.trainable = False
                            #print(f"Freezing {main_name}/{layer.ln_feats1j.name}")
                            #layer.ln_feats1j.trainable = False 
                            #print(f"Freezing {main_name}/{layer.bn_feats_1j.name}")
                            #layer.bn_feats_1j.trainable = False

                            print(f"Freezing {main_name}/{layer.hm1_output.name}")
                            layer.hm1_output.trainable = False
                            print(f"Freezing {main_name}/{layer.hm2_output.name}")
                            layer.hm2_output.trainable = False
                            print(f"Freezing {main_name}/{layer.features_hm2.name}")
                            layer.features_hm2.trainable = False
                            print(f"Freezing {main_name}/{layer.residual_2j.name}")
                            layer.residual_2j.trainable = False
                        #else:

                elif self.config.loading_style == "Partial_Train_F2S":
                    print(">>>>>>>>>> [LOADING] PARTIAL TRAINING FOR F2S (Top-down) <<<<<<<<<<<<<<")
                    for layer in model.layers:
                        if isinstance(layer,DownSamplingLayer):
                            print(f"Freezing {layer.name}")
                            layer.trainable = False
                        elif isinstance(layer,HourglassLayer):
                            main_name = layer.name
                            # Train only the main hourglass
                            for _,val in layer.layer_list.items():
                                val["up_1"].trainable = False # Freeze Skip layers
                                print("Freezing {}/{}".format(main_name,val["up_1"].name))
                                val["low_1"].trainable = False # Freeze S2F layers
                                print("Freezing {}/{}".format(main_name,val["low_1"].name))
                                val["low_3"].trainable = True # Train F2S layers
                                #val["low_3"].residual_blocks[0].conv_block.trainable = False # Freeze the ConvBlock
                                
                                #val["low_3"].residual_blocks[0].alpha.trainable = False
                            print(f"Freezing {main_name}/{layer.residual_brc.name}")
                            layer.residual_brc.trainable = False
                            print(f"Freezing {main_name}/{layer.merge_feats_main.name}")
                            layer.merge_feats_main.trainable = False
                            print(f"Freezing {main_name}/{layer.merge_feats_1j.name}")
                            layer.merge_feats_1j.trainable = False
                            
                            #print(f"Freezing {main_name}/{layer.ln_inputs.name}")
                            #layer.ln_inputs.trainable = False
                            #print(f"Freezing {main_name}/{layer.ln_main.name}")
                            #layer.ln_main.trainable = False
                            #print(f"Freezing {main_name}/{layer.ln_feats1j.name}")
                            #layer.ln_feats1j.trainable = False 
                            #print(f"Freezing {main_name}/{layer.bn_feats_1j.name}")
                            #layer.bn_feats_1j.trainable = False
                            
                            print(f"Freezing {main_name}/{layer.hm1_output.name}")
                            layer.hm1_output.trainable = False
                            print(f"Freezing {main_name}/{layer.hm2_output.name}")
                            layer.hm2_output.trainable = False
                            print(f"Freezing {main_name}/{layer.features_hm2.name}")
                            layer.features_hm2.trainable = False
                            print(f"Freezing {main_name}/{layer.residual_2j.name}")
                            layer.residual_2j.trainable = False
                elif self.config.loading_style == "Partial_Train_S2FF2S":
                    print(">>>>>>>>>> [LOADING] PARTIAL TRAINING FOR S2F - F2S <<<<<<<<<<<<<<")
                    for layer in model.layers:
                        if isinstance(layer,DownSamplingLayer):
                            print(f"Freezing {layer.name}")
                            layer.trainable = False
                        elif isinstance(layer,HourglassLayer):
                            main_name = layer.name
                            # Train only the main hourglass
                            for _,val in layer.layer_list.items():
                                val["up_1"].trainable = False # Freeze Skip layers
                                print("Freezing {}/{}".format(main_name,val["up_1"].name))
                                val["low_1"].trainable = True # Freeze S2F layers
                                val["low_1"].residual_blocks[0].conv_block.trainable = False # Freeze the ConvBlock
                                #print("Freezing {}/{}".format(main_name,val["low_1"].name))
                                val["low_3"].trainable = True # Train F2S layers
                                val["low_3"].residual_blocks[0].conv_block.trainable = False # Freeze the ConvBlock
                                #val["low_3"].residual_blocks[0].alpha.trainable = False
                            print(f"Freezing {main_name}/{layer.residual_brc.name}")
                            layer.residual_brc.trainable = False
                            print(f"Freezing {main_name}/{layer.merge_feats_main.name}")
                            layer.merge_feats_main.trainable = False
                            print(f"Freezing {main_name}/{layer.merge_feats_1j.name}")
                            layer.merge_feats_1j.trainable = False
                            
                            print(f"Freezing {main_name}/{layer.hm1_output.name}")
                            layer.hm1_output.trainable = False
                            print(f"Freezing {main_name}/{layer.hm2_output.name}")
                            layer.hm2_output.trainable = False
                            print(f"Freezing {main_name}/{layer.features_hm2.name}")
                            layer.features_hm2.trainable = False
                            print(f"Freezing {main_name}/{layer.residual_2j.name}")
                            layer.residual_2j.trainable = False
                elif self.config.loading_style == "Partial_Train_Skip":
                    print(">>>>>>>>>> [LOADING] PARTIAL TRAINING FOR SKIP LAYERS <<<<<<<<<<<<<<")
                    for layer in model.layers:
                        if isinstance(layer,DownSamplingLayer):
                            print(f"Freezing {layer.name}")
                            layer.trainable = False
                        elif isinstance(layer,HourglassLayer):
                            main_name = layer.name
                            # Train only the main hourglass
                            for _,val in layer.layer_list.items():
                                val["up_1"].trainable = False # Train Skip layers
                                val["up_1"].residual_blocks[0].conv_block.trainable = False # Freeze the ConvBlock
                                #val["up_1"].residual_blocks[0].alpha.trainable = False
                                val["low_1"].trainable = False # Freeze S2F layers
                                print("Freezing {}/{}".format(main_name,val["low_1"].name))
                                val["low_3"].trainable = True # Freeze F2S layers
                                print("Freezing {}/{}".format(main_name,val["low_3"].name))
                            print(f"Freezing {main_name}/{layer.residual_brc.name}")
                            layer.residual_brc.trainable = True
                            print(f"Freezing {main_name}/{layer.merge_feats_main.name}")
                            layer.merge_feats_main.trainable = False
                            print(f"Freezing {main_name}/{layer.merge_feats_1j.name}")
                            layer.merge_feats_1j.trainable = False
                            
                            #print(f"Freezing {main_name}/{layer.ln_inputs.name}")
                            #layer.ln_inputs.trainable = False
                            #print(f"Freezing {main_name}/{layer.ln_main.name}")
                            #layer.ln_main.trainable = False
                            #print(f"Freezing {main_name}/{layer.ln_feats1j.name}")
                            #layer.ln_feats1j.trainable = False 
                            #print(f"Freezing {main_name}/{layer.bn_feats_1j.name}")
                            #layer.bn_feats_1j.trainable = False
                            
                            print(f"Freezing {main_name}/{layer.hm1_output.name}")
                            layer.hm1_output.trainable = False
                            print(f"Freezing {main_name}/{layer.hm2_output.name}")
                            layer.hm2_output.trainable = False
                            print(f"Freezing {main_name}/{layer.features_hm2.name}")
                            layer.features_hm2.trainable = False
                            print(f"Freezing {main_name}/{layer.residual_2j.name}")
                            layer.residual_2j.trainable = False   
                elif self.config.loading_style == "Full_Train":
                    print(">>>>>>>>>> [LOADING] FULL TRAINING <<<<<<<<<<<<<<")               
                        #for _l in HourglassLayer.layers:
                        #    print(f"XX {layer.name}")
                        #layer.trainable = False
                elif self.config.loading_style == "Partial_Downsampling":
                    print(">>>>>>> [LOADING] PARTIAL TRAINING DOWNSAMPLING <<<<<<<<<")
                    for layer in model.layers:
                        if isinstance(layer,DownSamplingLayer):
                            layer.trainable = True
                        else:
                            print(f"Freezing {layer.name}")
                            layer.trainable = False
                elif self.config.loading_style == "Partial_Downsampling_frozen":
                    print(">>>>>>> [LOADING] PARTIAL TRAINING DOWNSAMPLING FROZEN <<<<<<<<<")
                    for layer in model.layers:
                        if isinstance(layer,DownSamplingLayerLora):
                            print(f"Freezing {layer.name}")
                            layer.trainable = False
                        else:
                            layer.trainable = True
                elif self.config.loading_style == "Partial_Joint_Scheme":
                    print(">>>>>>>>>> [LOADING] PARTIAL TRAINING FOR CHANGING JOINT SCHEME <<<<<<<<<<<<<<")
                    for layer in model.layers:
                        if isinstance(layer,DownSamplingLayer):
                            print(f"Freezing {layer.name}")
                            layer.trainable = False
                        elif isinstance(layer,HourglassLayer):
                            main_name = layer.name
                            # Train only the main hourglass
                            for _,val in layer.layer_list.items():
                                for _,v in val.items():
                                    if isinstance(v,Layer):
                                        #print(f"Freezing {main_name}/{v.name}")
                                        v.trainable = False
                            print(f"Freezing {main_name}/{layer.residual_brc.name}")
                            layer.residual_brc.trainable = False
                            print(f"Freezing {main_name}/{layer.merge_feats_main.name}")
                            layer.merge_feats_main.trainable = False
                            print(f"Freezing {main_name}/{layer.features_hm2.name}")
                            layer.features_hm2.trainable = False
                            print(f"Freezing {main_name}/{layer.residual_2j.name}")
                            layer.residual_2j.trainable = False
                model.summary()
                self._model = model
            # Link Input Shape to Model
            self._output = model(inputs=input_tensor, *args, **kwargs)
        else:
            model = self._build_model_as_layer(*args, **kwargs)


# endregion
