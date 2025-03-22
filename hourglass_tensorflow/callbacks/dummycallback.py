import tensorflow as tf
import numpy as np
from hourglass_tensorflow.models.hourglass import HourglassModel
from hourglass_tensorflow.metrics.correct_keypoints import PercentageOfCorrectKeypoints
from hourglass_tensorflow.metrics.distance import OverallMeanDistance,SoftargmaxMeanDist
from hourglass_tensorflow.losses import MAE_custom
from time import sleep
import keras
from keras.callbacks import Callback
import keras.models
import json

from hourglass_tensorflow.layers.conv_block import ConvBlockLayer
from hourglass_tensorflow.layers.hourglass_Beta import HourglassLayer
from hourglass_tensorflow.layers.residual import ResidualLayer,ResidualLayerIn,ResidualBlock,ResidualBlockIn
from hourglass_tensorflow.layers.skip import SkipLayer
#from hourglass_tensorflow.layers.residual_with_attention import ResidualLayerAttention
#from hourglass_tensorflow.layers.residual_with_attention_spatial import ResidualLayerAttentionSpatial
from hourglass_tensorflow.layers.downsampling import DownSamplingLayer
from hourglass_tensorflow.layers.hourglass_Beta import HourglassLayer
from hourglass_tensorflow.layers.batch_norm_relu_conv import BatchNormReluConvLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer
from hourglass_tensorflow.layers.dummy_layers import zeroLayer,IdentityLayer
from hourglass_tensorflow.layers.linear_projection import LinearProjection
from hourglass_tensorflow.utils.loaders.weight_loader import recursive_weight_transfer
from hourglass_tensorflow.utils.loaders.model_loader import load_basemodel_weights


class DummyCallback(Callback):
    def __init__(self, x_val=None):
        super(DummyCallback, self).__init__()
        self.input_data = x_val
    
    def on_epoch_end(self, epoch, logs=None):
        if int(epoch) % 4 ==0 and False:
            print(f"Executing dummy callback at epoch: {epoch}")
            data1 = self.input_data.map(lambda imgs:1.0*imgs)
            data2 = self.input_data.map(lambda imgs:1.0*imgs)
            """
            for k,(d1,d2) in enumerate(zip(data1,data2)):
                data_match = np.allclose(d1.numpy(), d2.numpy())
                if not data_match:
                    print(f"DATA MISMATCH --- {k}")
                else:
                    print(f"DATA MATCH --- {k}")
            """
            # Save the model and check consistency in the results
            #keras.config.enable_unsafe_deserialization()
            _ = self.model(tf.ones((1, 256, 256, 1)), training=True)
            self.model.trainable = True
            self.model.save("data/dummymodel.keras")
            print(json.dumps(self.model.get_config(), indent=1)) 
            #self.model.save_weights("data/baseline.weights.h5")
            sleep(2.0)
            #self.model.trainable = False  # Set to inference mode (no training layers active)
            print("LOADING DUMMY MODEL")
            cfgmodel = self.model.get_config()
            #print(cfgmodel)
            #dummymodel = keras.models.clone_model(self.model)  # Ensure identical structure
            #dummymodel = HourglassModel(**cfgmodel)
            #dummymodel.build((None, 256, 256, 1))
            """
            dummymodel = keras.models.load_model("data/dummymodel.keras", #"data/model_t/myModel_SLP_fABC10_2j.keras", #,compile=False)
                                       custom_objects= {#"RatioCorrectKeypoints":RatioCorrectKeypoints
                                            "HourglassModel": HourglassModel,
                                            "HourglassLayer": HourglassLayer,
                                            "ConvBlockLayer": ConvBlockLayer,
                                            "ResidualLayer": ResidualLayer,
                                            "ResidualLayerIn": ResidualLayerIn,
                                            "ResidualBlock": ResidualBlock,
                                            "ResidualBlockIn": ResidualBlockIn,
                                            "SkipLayer": SkipLayer,
                                            "DownsamplingLayer": DownSamplingLayer,
                                            "BatchNormReluConvLayer": BatchNormReluConvLayer,
                                            "ConvBatchNormReluLayer": ConvBatchNormReluLayer,
                                            "IdentityLayer": IdentityLayer,
                                            "zeroLayer": zeroLayer,
                                            "LinearProjection": LinearProjection,
                                            "PercentageOfCorrectKeypoints":PercentageOfCorrectKeypoints,
                                            "MAE_custom":MAE_custom,
                                            "OverallMeanDistance":OverallMeanDistance,
                                            "SoftargmaxMeanDist":SoftargmaxMeanDist},compile=False)
            
            """
            dummymodel = load_basemodel_weights(self.model,"data/dummymodel.keras",compile=False)
            #dummymodel = load_basemodel_weights(self.model,"data/model_t/myModel_SLP_fABC10_2j.keras",compile=False)
            #lazyInput = tf.ones(shape=(160,256,256,1),dtype=tf.float32)
            #lazydataset  = tf.data.Dataset.from_tensor_slices(lazyInput).batch(40)
            #dummymodel.predict(lazydataset)
            #matched, skipped = recursive_weight_transfer(dummymodel,self.model)
            #print(f"\nSummary: {matched} matched | {skipped} skipped")
            weights_model = self.model.get_weights()
            #dummymodel.load_weights("data/baseline.weights.h5",skip_mismatch=True)
            #dummymodel.set_weights(weights_model)
            weights_dummymodel = dummymodel.get_weights()
            print("===== MODEL SUMMARY ======")
            self.model.summary()
            print("===== DUMMYMODEL SUMMARY ======")
            dummymodel.summary()
            for layer in dummymodel.layers:
                if isinstance(layer, keras.layers.BatchNormalization):
                    print(f"Layer {layer.name} - Gamma: {layer.gamma.numpy()}")
                    print(f"Layer {layer.name} - Beta: {layer.beta.numpy()}")
                    print(f"Layer {layer.name} - Moving Mean: {layer.moving_mean.numpy()}")
                    print(f"Layer {layer.name} - Moving Variance: {layer.moving_variance.numpy()}")
                else:
                    #for _layer in layer.submodules:
                    print(f" --- Layer {layer.name}: {layer.get_config()}")
            #"""
            if(False):
                try:
                    # Check if the weights match and print more detailed information
                    for idx, (w1, w2) in enumerate(zip(weights_model, weights_dummymodel)):
                        # Check if the weights are the same using np.allclose
                        #layer_name = self.model.layers[idx].name
                        weights_match = np.allclose(w1, w2,atol=1e-6)
                        
                        # Display detailed information
                        if weights_match:
                            print(f"[Layer: {idx}] Weights match.")
                        else:
                            # Optionally, display the difference (for debugging purposes)
                            difference = np.abs(w1 - w2)
                            max_diff = np.mean(difference)
                            print(f"[Layer: {idx}] Weights do not match. || {max_diff}")
                            #"""
                except:
                    print("Could not compare the weights")
                    print(w1,type(w1),w2,type(w2))
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
            #"""
            
            #"""
            dummymodel.trainable = False
            self.model.trainable = False
            y_pred0 = self.model.predict(self.input_data)
            y_pred1 = dummymodel.predict(self.input_data)
            error = tf.math.abs(y_pred0-y_pred1)
            error2 = tf.reduce_mean(error,axis=[1,2,3,4])
            error3 = tf.reduce_mean(error2)
            print("Error prediction::", error3)
            self.model.trainable = True # Set to inference mode (no training layers active
    def on_train_begin(self, logs=None):
        # Initialization of the best metric value
        self.best = np.float32(-float('inf'))
        self.wait = 0
        self.lr_reduced = False