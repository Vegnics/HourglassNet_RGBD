import keras
import tensorflow as tf
from hourglass_tensorflow.metrics.correct_keypoints import PercentageOfCorrectKeypoints
from hourglass_tensorflow.metrics.distance import OverallMeanDistance,SoftargmaxMeanDist
from hourglass_tensorflow.losses import MAE_custom

"""
from hourglass_tensorflow.models.hourglass import HourglassModel
from hourglass_tensorflow.layers.conv_block import ConvBlockLayer
from hourglass_tensorflow.layers.hourglass_Beta2 import HourglassLayer
from hourglass_tensorflow.layers.residual_exp2 import ResidualLayer,ResidualLayerIn,ResidualBlock,ResidualBlockIn
from hourglass_tensorflow.layers.downsampling import DownSamplingLayer
"""

#"""
from hourglass_tensorflow.models.hourglass_lora import HourglassModelLora 
from hourglass_tensorflow.layers.conv_block_lora import ConvBlockLoRALayer 
from hourglass_tensorflow.layers.hourglass_lora import HourglassLayerLora 
from hourglass_tensorflow.layers.residual_lora import ResidualLayer,ResidualLayerIn,ResidualBlock,ResidualBlockIn
from hourglass_tensorflow.layers.downsampling_lora import DownSamplingLayerLora 
#"""


from hourglass_tensorflow.layers.skip import SkipLayer
from hourglass_tensorflow.layers.batch_norm_relu_conv import BatchNormReluConvLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer
from hourglass_tensorflow.layers.dummy_layers import zeroLayer,IdentityLayer
from hourglass_tensorflow.layers.linear_projection import LinearProjectionLoRA as LinearProjection
from hourglass_tensorflow.utils.loaders.weight_loader import recursive_weight_transfer


def load_wrapped_model(fpath: str = None, compile:bool=False):
    wmodel = keras.models.load_model(fpath,custom_objects= {
                                            "HourglassModelLora": HourglassModelLora,
                                            "HourglassLayerLora": HourglassLayerLora,
                                            "ConvBlockLoRALayer": ConvBlockLoRALayer,
                                            "ResidualLayer": ResidualLayer,
                                            "ResidualLayerIn": ResidualLayerIn,
                                            "ResidualBlock": ResidualBlock,
                                            "ResidualBlockIn": ResidualBlockIn,
                                            "SkipLayer": SkipLayer,
                                            "DownSamplingLayerLora": DownSamplingLayerLora,
                                            "BatchNormReluConvLayer": BatchNormReluConvLayer,
                                            "ConvBatchNormReluLayer": ConvBatchNormReluLayer,
                                            "IdentityLayer": IdentityLayer,
                                            "zeroLayer": zeroLayer,
                                            "LinearProjection": LinearProjection,
                                            "PercentageOfCorrectKeypoints":PercentageOfCorrectKeypoints,
                                            "MAE_custom":MAE_custom,
                                            "OverallMeanDistance":OverallMeanDistance,
                                            "SoftargmaxMeanDist":SoftargmaxMeanDist},compile=compile)
    return wmodel

def load_basemodel_weights(ftmodel:keras.models.Model = None, basepath: str = None, compile:bool=False):
    basemodel = keras.models.load_model(basepath,
                custom_objects= {
                    "HourglassModelLora": HourglassModelLora,
                    "HourglassLayerLora": HourglassLayerLora,
                    "ConvBlockLoRALayer": ConvBlockLoRALayer,
                    "ResidualLayer": ResidualLayer,
                    "ResidualLayerIn": ResidualLayerIn,
                    "ResidualBlock": ResidualBlock,
                    "ResidualBlockIn": ResidualBlockIn,
                    "SkipLayer": SkipLayer,
                    "DownSamplingLayerLora": DownSamplingLayerLora,
                    "BatchNormReluConvLayer": BatchNormReluConvLayer,
                    "ConvBatchNormReluLayer": ConvBatchNormReluLayer,
                    "IdentityLayer": IdentityLayer,
                    "zeroLayer": zeroLayer,
                    "LinearProjection": LinearProjection,
                    "PercentageOfCorrectKeypoints":PercentageOfCorrectKeypoints,
                    "MAE_custom":MAE_custom,
                    "OverallMeanDistance":OverallMeanDistance,
                    "SoftargmaxMeanDist":SoftargmaxMeanDist},compile=compile)
    #lazyInput = tf.ones(shape=(160,256,256,1),dtype=tf.float32)
    #lazydataset  = tf.data.Dataset.from_tensor_slices(lazyInput).batch(40)
    #basemodel.predict(lazydataset)
    matched, skipped = recursive_weight_transfer(basemodel,ftmodel)
    print(f"\nSummary: {matched} matched | {skipped} skipped")
    return basemodel