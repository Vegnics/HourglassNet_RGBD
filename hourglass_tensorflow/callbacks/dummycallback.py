import tensorflow as tf
import numpy as np
from hourglass_tensorflow.models.hourglass import HourglassModel
from hourglass_tensorflow.metrics.correct_keypoints import PercentageOfCorrectKeypoints
from hourglass_tensorflow.metrics.distance import OverallMeanDistance,SoftargmaxMeanDist
from hourglass_tensorflow.losses import MAE_custom

class DummyCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val=None,dummymodel=None):
        super(DummyCallback, self).__init__()
        self.input_data = x_val
        self.dummymodel = dummymodel
    
    def on_epoch_end(self, epoch, logs=None):
        if int(epoch) % 4 ==0:
            print(f"Executing dummy callback at epoch: {epoch}")
            # Save the model and check consistency in the results
            self.model.save("data/dummy_model2.keras")
            #self.model.trainable = False  # Set to inference mode (no training layers active)
            print("LOADING DUMMY MODEL")
            dummymodel = tf.keras.models.load_model("data/dummy_model2.keras",compile=False)
            """,
                                        custom_objects= {#"RatioCorrectKeypoints":RatioCorrectKeypoints
                                            "HourglassModel": HourglassModel,
                                            "PercentageOfCorrectKeypoints":PercentageOfCorrectKeypoints,
                                            "MAE_custom":MAE_custom,
                                            "OverallMeanDistance":OverallMeanDistance,
                                            "SoftargmaxMeanDist":SoftargmaxMeanDist},compile=False)
            """
            #self.dummymodel.build(input_shape=(None, 256, 256, 1))
            lazyInput = tf.zeros(shape=(80,256,256,1),dtype=tf.float32)
            dummymodel.trainable = True
            dummymodel.predict(lazyInput)
            weights_model = self.model.get_weights()
            weights_dummymodel = dummymodel.get_weights()
            print("===== MODEL SUMMARY ======")
            self.model.summary()
            print("===== DUMMYMODEL SUMMARY ======")
            dummymodel.summary()
            """
            try:
                # Check if the weights match and print more detailed information
                for idx, (w1, w2) in enumerate(zip(weights_model, weights_dummymodel)):
                    # Check if the weights are the same using np.allclose
                    #layer_name = self.model.layers[idx].name
                    weights_match = np.allclose(w1, w2)
                    
                    # Display detailed information
                    if weights_match:
                        print(f"[Layer: {idx}] Weights match.")
                    else:
                        print(f"[Layer: {idx}] Weights do not match.")
                        # Optionally, display the difference (for debugging purposes)
                        difference = np.abs(w1 - w2)
                        max_diff = np.max(difference)
                        """
            """
                        print(f"  Max difference in weights: {max_diff:.5f}")

                        # Show a few sample differences for better visualization (optional)
                        if max_diff > 1e-6:  # Adjust this threshold if necessary
                            print(f"  Sample differences in weights (first 5 elements):")
                            print(f"    Model weights: {w1.flatten()[:5]}")
                            print(f"    Dummy model weights: {w2.flatten()[:5]}")
                            print(f"    Difference: {difference.flatten()[:5]}")
                        print("\n")  # Add a blank line for better readability
                        """
            """
            except:
                print("Could not compare the weights")
                print(w1,type(w1),w2,type(w2))
            """
            #self.model.trainable = False
            #self.dummymodel.trainable = False
            y_pred0 = self.model.predict(self.input_data)
            y_pred1 = dummymodel.predict(self.input_data)
            error = tf.math.square(y_pred0-y_pred1)
            error2 = tf.reduce_mean(error,axis=[1,2,3,4])
            error3 = tf.reduce_mean(error2)
            print("Error prediction::", error3)
            self.model.trainable = True # Set to inference mode (no training layers active
            logs['LR'] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    def on_train_begin(self, logs=None):
        # Initialization of the best metric value
        self.best = np.float32(-float('inf'))
        self.wait = 0
        self.lr_reduced = False