from abc import abstractmethod
from typing import List
from typing import Union
from typing import TypeVar

import tensorflow as tf
from keras.losses import Loss
from keras.models import Model
from keras.metrics import Metric
from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule
#from keras.optimizers.schedules import LearningRateSchedule

from hourglass_tensorflow.types.config import HTFTestConfig
from hourglass_tensorflow.types.config import HTFObjectReference
from hourglass_tensorflow.handlers.meta import _HTFHandler
from hourglass_tensorflow.utils.tf import tf_dynamic_matrix_argmax

# region Abstract Class

R = TypeVar("R")


class _HTFTestHandler(_HTFHandler):
    def __init__(
        self,
        config: HTFTestConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config=config, *args, **kwargs)
        #self._metrics: List[HTFObjectReference[Metric]] = None
        self.init_handler()

    @property
    def config(self) -> HTFTestConfig:
        return self._config

    @abstractmethod
    def compile(self, model: Model, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def test(
        self,
        model: Model,
        test_dataset: tf.data.Dataset = None,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        #self.compile(*args, **kwargs)
        self.test(*args, **kwargs)


# enregion

"""Model = tf.keras.models.load_model("data/model_t/myModel_SLP_e",
                           custom_objects= {"RatioCorrectKeypoints":RatioCorrectKeypoints
                                            ,"PercentageOfCorrectKeypoints":PercentageOfCorrectKeypoints,
                                            "MAE_custom":MAE_custom})
"""
                                            
# region Handler


class HTFTestHandler(_HTFTestHandler):
    def _instantiate(self, obj: HTFObjectReference[R], **kwargs) -> R:
        if isinstance(obj, HTFObjectReference):
            return obj.init(**kwargs)
        else:
            return obj

    def init_handler(self, *args, **kwargs) -> None:
        self._batch_size = self.config.batch_size
        """
        self._learning_rate = self._instantiate(self.config.learning_rate)
        self._loss = self._instantiate(self.config.loss)
        self._optimizer = self._instantiate(
            self.config.optimizer, learning_rate=self._learning_rate
        )
        self._metrics = [obj.init() for obj in self.config.metrics]
        self._callbacks = [obj.init() for obj in self.config.callbacks]
        """

    def compile(self, model: Model, *args, **kwargs) -> None:
        model.compile(optimizer=self._optimizer, metrics=self._metrics, loss=self._loss)
    
    def _apply_batch(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        if isinstance(dataset, tf.data.Dataset):
            print("Batch size ",self._batch_size)
            return dataset.batch(self._batch_size)
        
    def _get_second_max_single(self, kpnt: tf.Tensor, heatmaps: tf.Tensor):
        pnt = 1*kpnt
        x = pnt[0]
        y = pnt[1]
        n = 1.0*heatmaps[y-1:y+2,x-1:x+2]
        if n.shape[0]==3 and n.shape[1]==3:
            m = tf.convert_to_tensor([[1,1,1],
                                [1,0,1],
                                [1,1,1]],dtype=tf.float32)
            
            #print("SINGLE ",m.shape, n.shape)
            _n = n*m
            pnt = tf.argmax(tf.reshape(_n,(-1,1)))
            dx = tf.cast(pnt[0]%3-1,dtype=tf.int32)
            dy = tf.cast(pnt[0]//3-1,dtype=tf.int32)

            hmvals = tf.convert_to_tensor([0.4,heatmaps[y+dy,x+dx]]) 
            maxidx = tf.argmax(hmvals)
            deltas = tf.convert_to_tensor([[0,0],[dx,dy]])
            return (deltas[maxidx,:],0)
        else:
            return (tf.convert_to_tensor([0,0]),0)
        
    def get_second_max(self, kpntLocs: tf.Tensor,heatmaps: tf.Tensor):
        # keypoints: Cx2
        #print("get second max :", kpntLocs.shape, heatmaps.shape)
        heatmapsRS = tf.transpose(heatmaps,perm=[2,0,1])
        auxiliars = tf.map_fn(
            fn= lambda elms: self._get_second_max_single(elms[0],elms[1]),#heatmaps[:,:,elms[1][0]]),
            elems=(
                kpntLocs,
                heatmapsRS 
                #tf.reshape(tf.range(0,14,1),(14,1))
            ),
            parallel_iterations=10,
        )
        #print("AUXILIARS ",auxiliars)
        return auxiliars

    def _refine_pred(self,rjoint,auxiliar,bbox):
        pnt = tf.cast(rjoint,tf.float32)
        dpnt = tf.cast(auxiliar,tf.float32)
        
        pad = bbox[2,0:2]
        Hb = bbox[1,1] - bbox[0,1]
        Wb = bbox[1,0] - bbox[0,0]
        N = tf.cast(tf.math.maximum(Wb,Hb),tf.float32)
        padx = tf.cast(pad[0]*64,tf.float32)/N
        pady = tf.cast(pad[1]*64,tf.float32)/N
        rx = tf.cast((N/64.0)*(pnt[:,0] + 0.25*dpnt[:,0] - padx),tf.int32)+ tf.cast(bbox[0,0],tf.int32)
        ry = tf.cast((N/64.0)*(pnt[:,1] + 0.25*dpnt[:,1] - pady),tf.int32)+ tf.cast(bbox[0,1],tf.int32)
        out = tf.convert_to_tensor([rx,ry])
        out = tf.transpose(out,perm=[1,0])
        #print(out,bbox.shape)
        return out

    def refine_predictions(self,heatmaps: tf.Tensor, bboxes):
        # rjoints: NxCx2
        # bboxes: Nx3x2
        joints = tf_dynamic_matrix_argmax(
            heatmaps[:,:,:,0:14],
            intermediate_supervision=False,
            keepdims=True,
        )



        auxiliars = tf.map_fn(
            fn=lambda jointAndHM: self.get_second_max(jointAndHM[0],jointAndHM[1]),
            elems=(
                joints,
                heatmaps[:,:,:,0:14]),
            parallel_iterations=10,
        )

        auxiliars = auxiliars[0]
        print(auxiliars.shape)


        RJoints = tf.map_fn(
            fn=lambda jointNbox: self._refine_pred(jointNbox[0],jointNbox[1],jointNbox[2]),
            elems=(
                joints,
                auxiliars,
                bboxes
            ),
            fn_output_signature=tf.int32,
            parallel_iterations=10,
        )
        #print(RJoints[0].shape)
        return  RJoints 


    def test(
        self,
        model: Model,
        test_dataset: tf.data.Dataset = None,
        *args,
        **kwargs,
    ) -> None:
        with tf.device('/GPU:0'):
            #test_dataset = self._apply_batch(test_dataset)
            tds_card = 2400
            vds_card = 300
            #validation_dataset = validation_dataset.repeat(2)
            imgs_ds = test_dataset.map(lambda imgs,coords,bboxes: imgs)
            batch_test = self._apply_batch(imgs_ds) 
            #print("   ??????    >>>BATCH TRAIN: ",batch_train)
            #batch_validation = validation_dataset.batch(150)#self._apply_batch(validation_dataset)
            batch_num = batch_test.__len__()
            print("BATCH INFO :", batch_num.numpy().tolist())
            model.summary()

            hm_scale = tf.constant(2.1269474)
            #imgs = batch_test.map(lambda imgs,coords,bboxes: imgs).get_single_element()
            bboxes = test_dataset.map(lambda imgs,coords,bboxes: bboxes)#.get_single_element()
            #bboxes = bboxes.reduce(tf.zeros(shape=(1,3,2)),lambda x, y: tf.concat([x, expanduy], axis=0))
            #bboxes = tf.convert_to_tensor(list(bboxes.as_numpy_iterator()))
            #print("BBOXES:::", bboxes)
            bboxes = tf.convert_to_tensor(list(bboxes.as_numpy_iterator()))
            print("BBOXES: ", bboxes.shape)
            gtcoords = test_dataset.map(lambda imgs,coords,bboxes: coords)#.get_single_element()
            gtcoords = tf.convert_to_tensor(list(gtcoords.as_numpy_iterator()))
            print("GTCOORDS: ", gtcoords.shape)

            _preds = model.predict(x=batch_test) #NSHWC
            preds = _preds[:,-1,:,:,:] 
            #normpreds = tf.linalg.norm(preds,axis=[1,2])
            #normpreds = tf.expand_dims(normpreds,axis=1)
            #normpreds = tf.expand_dims(normpreds,axis=1)
            #preds = preds*hm_scale/(normpreds+0.00001)
            print(preds.shape)
            predcoords = self.refine_predictions(preds,bboxes)
            #predcoords = gtcoords + 5.0
            print("Refined predictions : ..")
            #print(predcoords,"\n",gtcoords)


            error = tf.cast(tf.cast(gtcoords,tf.float32) - tf.cast(predcoords,tf.float32), dtype=tf.dtypes.float32)
            distance = tf.norm(error, ord=2, axis=-1) #NxC
            # We compute the norm of the reference limb from the ground truth
            reference_limb_error = tf.cast(
                gtcoords[:, 13, :]
                - gtcoords[:, 12, :],
                dtype=tf.float32,
            )# Nx2
            # Compute the reference distance (It could be the head distance, or torso distance)
            
            #mask_tensor = 2.0*(1.0-vis)*tf.constant(32.0)
            #distance = distance + mask_tensor


            reference_distance = tf.norm(reference_limb_error, ord=2, axis=-1) #N
            #max_ref = tf.reduce_max(reference_distance)
            
            reference_distance = tf.expand_dims(reference_distance,axis=1) #Nx1
            # We apply the thresholding condition
            condition = tf.cast(tf.math.less(distance,reference_distance * 0.5),
                                    dtype=tf.float32)
            correct_keypoints = tf.reduce_sum(condition)
            print(correct_keypoints,"out of :",14*3105)

# endregion