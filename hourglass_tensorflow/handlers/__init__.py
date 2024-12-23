from typing import Union
from typing import TypeVar
from matplotlib import pyplot as plt
import numpy as np
import cv2

import tensorflow as tf
from hourglass_tensorflow.utils import ObjectLogger
from hourglass_tensorflow.utils import BadConfigurationError
from hourglass_tensorflow.types.config import HTFConfig
from hourglass_tensorflow.types.config import HTFMetadata
from hourglass_tensorflow.types.config import HTFConfigMode
from hourglass_tensorflow.types.config import HTFConfigField
from hourglass_tensorflow.types.config import HTFConfigParser
from hourglass_tensorflow.types.config import HTFObjectReference
from hourglass_tensorflow.handlers.data import HTFDataHandler
from hourglass_tensorflow.handlers.meta import _HTFHandler
from hourglass_tensorflow.handlers.model import HTFModelHandler
from hourglass_tensorflow.handlers.train import HTFTrainHandler
from hourglass_tensorflow.handlers.test import HTFTestHandler
from hourglass_tensorflow.handlers.dataset import HTFDatasetHandler
from hourglass_tensorflow.utils.tf import tf_matrix_argmax,tf_batch_matrix_argmax

T = TypeVar("T")


class HTFManager(ObjectLogger):
    def __init__(self, filename: str, verbose: bool = True, *args, **kwargs) -> None:
        super().__init__(verbose, *args, **kwargs)
        self._config_file = filename
        #print(list(HTFConfigParser.parse(filename=filename, verbose=verbose).model_dump().keys()))
        self._config = HTFConfig.model_validate(
            HTFConfigParser.parse(filename=filename, verbose=verbose).model_dump()
        )
        self._metadata = HTFMetadata()

    @property
    def config(self) -> HTFConfig:
        return self._config

    @property
    def mode(self) -> HTFConfigMode:
        return self.config.mode

    @property
    def VALIDATION_RULES(self):
        return {
            HTFConfigMode.TRAIN: [],
            HTFConfigMode.TEST: [],
            HTFConfigMode.INFERENCE: [],
            HTFConfigMode.SERVER: [],
        }

    @property
    def metadata(self) -> HTFMetadata:
        return self._metadata

    def _import_object(
        self,
        obj: HTFObjectReference[T],
        config: HTFConfigField,
        metadata: HTFMetadata,
        *args,
        **kwargs
    ) -> Union[T, _HTFHandler]:
        instance = obj.init(config=config, metadata=metadata, *args, **kwargs)
        return instance

    def __call__(self, *args, **kwargs) -> None:

        if not all(self.VALIDATION_RULES[self.mode]):
            raise BadConfigurationError

        if self.mode == HTFConfigMode.TRAIN:
            self.train(*args, **kwargs)
        if self.mode == HTFConfigMode.TEST:
            self.test(*args, **kwargs)
        if self.mode == HTFConfigMode.INFERENCE:
            self.inference(*args, **kwargs)
        if self.mode == HTFConfigMode.SERVER:
            self.server(*args, **kwargs)

    def server(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def test(self, *args, **kwargs) -> None:
        obj_data: HTFObjectReference[HTFDataHandler] = self._config.data.object
        obj_dataset: HTFObjectReference[HTFDatasetHandler] = self._config.dataset.object
        obj_model: HTFObjectReference[HTFModelHandler] = self._config.model.object
        obj_test: HTFObjectReference[HTFTestHandler] = self._config.test.object
        # Launch Data Handler
        self.DATA = self._import_object(
            obj_data, config=self._config.data, metadata=self._metadata
        )
        data = self.DATA().get_data() # Call the HTFDataHandler and invoke get_data() method.
        # Launch Dataset Handler
        self.DATASET = self._import_object(
            obj_dataset,
            config=self._config.dataset,
            metadata=self._metadata,
            data=data,
        )
        self.DATASET()

        self.MODEL = self._import_object(
            obj_model,
            config=self._config.model,
            metadata=self._metadata,
        )
        self.MODEL()

        self.TEST = self._import_object(
            obj_test,
            config=self._config.test,
            metadata=self._metadata,
        )
        self.TEST(
            model=self.MODEL._model,
            test_dataset=self.DATASET._test_dataset
        )
        #raise NotImplementedError

    def inference(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def train(self, *args, **kwargs) -> None:
        # Unpack Objects
        obj_data: HTFObjectReference[HTFDataHandler] = self._config.data.object
        obj_dataset: HTFObjectReference[HTFDatasetHandler] = self._config.dataset.object
        obj_model: HTFObjectReference[HTFModelHandler] = self._config.model.object
        obj_train: HTFObjectReference[HTFTrainHandler] = self._config.train.object
        # Launch Data Handler
        self.DATA = self._import_object(
            obj_data, config=self._config.data, metadata=self._metadata
        )
        data = self.DATA().get_data() # Call the HTFDataHandler and invoke get_data() method.
        # Launch Dataset Handler
        self.DATASET = self._import_object(
            obj_dataset,
            config=self._config.dataset,
            metadata=self._metadata,
            data=data,
        )
        self.DATASET()
        """
        train_dataset=self.DATASET._train_dataset
        test_dataset=self.DATASET._test_dataset
        validation_dataset = self.DATASET._validation_dataset
        for data in train_dataset:
            img = data[0].numpy()
            #hmp = data[1]
            img_rgb = np.uint8(np.copy(img[:,:,0]))

            #img = np.copy(img[:,:,3])
            #print(img.shape,img.dtype)
            
            hmp = tf.expand_dims(data[1],axis=0)#.numpy()
            cntld = 0
            for i in range(14): #16
                val = np.max(hmp[0,-1,:,:,i])
                if val>0.3:
                    cntld += 1
            print(np.max(img),np.min(img),cntld)
            
            depth = img[:,:,0]#*255.0/3.5
            #depth = np.clip(depth,0,255)
            #depth= np.uint8(depth)
            #depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)
            #img = np.copy(depth)
            plt.imshow(depth,cmap="jet")
            plt.colorbar(location="left",orientation="vertical",cmap="jet")
            plt.show() 
            if cntld < 14 or True: 
                _maxargs = tf_batch_matrix_argmax(hmp[:,-1,:,:,:])
                for stg in [2]:#range(3):
                    print(f"Stage {stg}:::")
                    for i in range(14):
                        print(f"\t Landmark {i}")
                        #plt.imshow(hmp[0,stg,:,:,i],cmap="jet",vmin=0.0,vmax=1.0)
                        #plt.colorbar(location="left",orientation="vertical",cmap="jet")
                        #plt.show()
                        
                        
                        #print(_maxidx)
                        
                        #_maxidx = np.argmax(hmp[0,:,:,i])
                        
                        if i < 16:
                            _maxidx = _maxargs[0,i,:]
                            x = _maxidx[0].numpy()#_maxidx%64
                            y = _maxidx[1].numpy()#_maxidx//64
                            val = hmp[0,0,y,x,i]
                            if val>0.1:
                                center = (int(4*x),int(4*y))
                                #cv2.circle(img_rgb,center,5,(255,0,0),-1)
                                #cv2.putText(img,"{:2d}".format(i),center,cv2.FONT_HERSHEY_PLAIN,1.2,(0,0,255))
                #plt.imshow(img_rgb)
                #plt.show() 
        """
        
        #"""
        # Launch Model Handler
        self.MODEL = self._import_object(
            obj_model,
            config=self._config.model,
            metadata=self._metadata,
        )
        self.MODEL()
        # Launch Train Handler
        self.TRAIN = self._import_object(
            obj_train,
            config=self._config.train,
            metadata=self._metadata,
        )
        self.TRAIN(
            model=self.MODEL._model,
            train_dataset=self.DATASET._train_dataset,
            test_dataset=self.DATASET._test_dataset,
            validation_dataset=self.DATASET._validation_dataset,
        )
        #"""