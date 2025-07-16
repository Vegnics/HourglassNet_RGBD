import tensorflow as tf
import numpy as np
from keras.callbacks import Callback
import keras
from hourglass_tensorflow.layers.hourglass_Beta import HourglassLayer
from hourglass_tensorflow.layers.downsampling import DownSamplingLayer
from hourglass_tensorflow.models.hourglass import HourglassModel

class MetricReduceLROnPlateau(Callback):
    def __init__(self, monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6, verbose=1):
        super(MetricReduceLROnPlateau, self).__init__()
        self.monitor = monitor  # The metric you want to monitor
        self.factor = factor  # Factor by which the learning rate will be reduced
        self.patience = patience  # Number of epochs to wait before reducing
        self.min_lr = min_lr  # Minimum learning rate allowed
        self.verbose = verbose  # Verbosity mode
        self.wait = 0  # Wait counter
        self.best = np.float32(+float('inf'))  # Best value of the monitored metric
        self.lr_reduced = False  # Flag to track if LR was reduced
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose > 0:
                print(f"Warning: Metric '{self.monitor}' is not available. Available metrics are: {', '.join(list(logs.keys()))}")
            return
        # Save the model and check consistency in the results
        """
        if(epoch==-1): #will remove this block
            for layer in self.model.layers:
                        if isinstance(layer,DownSamplingLayer):
                            layer.trainable = False
                        elif isinstance(layer,HourglassLayer):
                            main_name = layer.name
                            # Train only the main hourglass
                            for _,val in layer.layer_list.items():
                                val["up_1"].trainable = False # Freeze the Skip layers
                                val["low_1"].trainable = True # Train S2F
                                val["low_1"].residual_blocks[0].conv_block.trainable = False # Freeze the ConvBlock
                                #val["low_1"].residual_blocks[0].alpha.trainable = False # Freeze the ConvBlock
                                val["low_3"].trainable = False # Freeze F2S
                            layer.residual_brc.trainable = False
                            layer.merge_feats_main.trainable = False
                            layer.merge_feats_1j.trainable = False
                            layer.ln_inputs.trainable = False
                            layer.ln_main.trainable = False
                            layer.ln_feats1j.trainable = False 
                            layer.hm1_output.trainable = False
                            layer.hm2_output.trainable = False
                            layer.features_hm2.trainable = False
                            layer.residual_2j.trainable = False
            self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics)
        elif(epoch==5):
            for layer in self.model.layers:
                        if isinstance(layer,DownSamplingLayer):
                            layer.trainable = False
                        else:
                             layer.trainable = True
            self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics)
            self.model.optimizer.learning_rate.assign(0.5e-5)
        elif(epoch==9):
            for layer in self.model.layers:
                        if isinstance(layer,DownSamplingLayer):
                            layer.trainable = False
                        elif isinstance(layer,HourglassLayer):
                            # Train only the main hourglass
                            for _,val in layer.layer_list.items():
                                val["up_1"].trainable = False # Freeze the Skip layers
                                val["low_1"].trainable = False # Train S2F
                                #val["low_1"].residual_blocks[0].alpha.trainable = False # Freeze the ConvBlock
                                val["low_3"].trainable = True # Freeze F2S
                                val["low_3"].residual_blocks[0].conv_block.trainable = False # Freeze the ConvBlock
            self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics)
            self.model.optimizer.learning_rate.assign(7.5e-5)
        elif(epoch==14):
            for layer in self.model.layers:
                        if isinstance(layer,DownSamplingLayer):
                            layer.trainable = False
                        else:
                             layer.trainable = True
            self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics)
            self.model.optimizer.learning_rate.assign(0.5e-5)
        """
        # Check if the metric has improved
        #if current > self.best:
        if current < self.best:
            if self.verbose > 0:
                print(f"\nThe value of {self.monitor} has improved.")
            self.best = current
            self.wait = 0
            self.lr_reduced = False
            #old_lr = np.float32(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))
            #self.model.optimizer.learning_rate.assign(old_lr)
            #print("LEARNING RATE",self.model.optimizer.learning_rate,type(self.model.optimizer.learning_rate))
        else:
            if self.verbose > 0:
                print(f"\nThe value of {self.monitor} DIDNT improve from {self.best}|| Attempt :{self.wait+1}.")
            self.wait += 1
            if self.wait >= self.patience:
                # Reduce the learning rate if the metric hasn't improved
                old_lr =  np.float32(self.model.optimizer.learning_rate.numpy())
                #np.float32(float(keras.backend.get_value(self.model.optimizer.learning_rate)))
                new_lr = np.float32(max(old_lr * self.factor, self.min_lr))
                
                if old_lr > new_lr:  # Only update if the new LR is lower
                    self.model.optimizer.learning_rate.assign(new_lr)
                    #tf.keras.backend.set_value(np.float32(self.model.optimizer.learning_rate), )
                    if self.verbose > 0:
                        print(f"\nEpoch {epoch+1}: {self.monitor} did not improve. Reducing learning rate to {new_lr}.")
                    self.lr_reduced = True
                self.wait = 0  # Reset wait counter
        #logs['LR'] = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        logs['LR'] = np.float32(self.model.optimizer.learning_rate.numpy())

    def on_train_begin(self, logs=None):
        # Initialization of the best metric value
        self.best = np.float32(+float('inf'))
        self.wait = 0
        self.lr_reduced = False