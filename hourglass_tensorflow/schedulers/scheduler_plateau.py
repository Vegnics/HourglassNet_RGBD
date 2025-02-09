import tensorflow as tf
import numpy as np

class MetricReduceLROnPlateau(tf.keras.callbacks.Callback):
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
                old_lr = np.float32(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))
                new_lr = np.float32(max(old_lr * self.factor, self.min_lr))
                
                if old_lr > new_lr:  # Only update if the new LR is lower
                    self.model.optimizer.learning_rate.assign(new_lr)
                    #tf.keras.backend.set_value(np.float32(self.model.optimizer.learning_rate), )
                    if self.verbose > 0:
                        print(f"\nEpoch {epoch+1}: {self.monitor} did not improve. Reducing learning rate to {new_lr}.")
                    self.lr_reduced = True
                self.wait = 0  # Reset wait counter
        logs['LR'] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    def on_train_begin(self, logs=None):
        # Initialization of the best metric value
        self.best = np.float32(+float('inf'))
        self.wait = 0
        self.lr_reduced = False