from numpy import shape
import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.saving import register_keras_serializable
from hourglass_tensorflow.layers.dummy_layers import IdentityLayer,zeroLayer
from hourglass_tensorflow.layers.sequential_layer import SequentialLayer

@register_keras_serializable(package="lBNReLuConvlora")
class BatchNormReluConvLayerWLoRA(Layer):
    """
    This layer performs Batch normalization, ReLu, and finally 2D convolution.
    """
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        activation: str = None,
        kernel_initializer: str = "glorot_uniform",
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        name: str = None,
        trainable: bool = True,
        use_relu: bool = True,
        normalized: bool = True,
        activate_lora: bool = None,
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store Config
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_relu = use_relu
        self.normalized = normalized
        self.activate_lora = activate_lora
        self.lora_a = None
        self.lora_b = None
        # Create Layers

        self.batch_norm = layers.BatchNormalization(
            axis=-1,
            momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable,
            name="BatchNorm_Identity",
        ) if self.normalized else IdentityLayer(name="BatchNorm_Identity")

        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            name="Conv2D",
            activation=None,
            kernel_initializer=self.kernel_initializer,
        )
        
        #lora path using a rank of 4
        self.lora = SequentialLayer(
            [       layers.LayerNormalization(axis=-1
                                              ,epsilon=1e-6,
                                              name="lora_ln"),
             
                    layers.DepthwiseConv2D(
                        kernel_size=self.kernel_size,
                        padding="same",
                        name="lora_dw"
                    ),

                    layers.Conv2D(
                        filters=4,
                        kernel_size=1,
                        strides=self.strides,
                        padding=self.padding,
                        name="lora_a",
                        activation="gelu",
                        #use_bias=False,
                        kernel_initializer="glorot_uniform",
                    ),

                    layers.Conv2D(
                        filters=self.filters,
                        kernel_size=1,
                        strides=self.strides,
                        padding=self.padding,
                        name="lora_b",
                        activation=None,
                        #use_bias=False,
                        kernel_initializer="zeros",
                    )
                ],
                name="lora_path",
                trainable=self.trainable
                ) if self.activate_lora else zeroLayer(self.filters,name="lora_path")

        self.relu = layers.ReLU(
            name="ReLU_identity",
        ) if self.use_relu else IdentityLayer(name="ReLU_identity")

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "use_relu": self.use_relu,
                "normalized": self.normalized,
                "activate_lora": self.activate_lora
            },
        }
    
    # Call function with LoRA weight merging 
    """
    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        x = self.batch_norm(inputs, training=training)
        x = self.relu(x)
        conv_kernel = self.conv.kernel
        if self.activate_lora and self.built and self.trainable:
            lora_k = tf.matmul(self.lora_a, self.lora_b) # shape=(kernel_size*kernel_size, filters, filters)
            #lora_k = tf.transpose(lora_k, perm=[1, 2, 0]) # shape=(filters,filters,kernel_size*kernel_size)
            #lora_k = tf.reshape(lora_k, shape=(self.filters, self.filters, self.kernel_size, self.kernel_size)) # shape=(filters,filters,kernel_size,kernel_size)
            #lora_k = tf.reshape(lora_k, shape=(self.kernel_size,self.kernel_size, self.in_channels, self.filters))
            conv_kernel = tf.add(conv_kernel, lora_k) # add LoRA weights to the convolution kernel
        #lora = self.lora(x)
        #lora = tf.nn.conv2d(x, lora_k, strides=[1, self.strides, self.strides, 1], padding=self.padding.upper())
        #y = self.conv(x)
        y = tf.nn.conv2d(x, conv_kernel, strides=[1, self.strides, self.strides, 1], padding=self.padding.upper())
        return y #+ lora
    """
    
    # Call method with parallel adapter
    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        x = self.batch_norm(inputs, training=training)
        x = self.relu(x)
        lora = self.lora(x)
        #lora = tf.nn.conv2d(x, lora_k, strides=[1, self.strides, self.strides, 1], padding=self.padding.upper())
        y = self.conv(x)
        return y + lora
    
    # Build method with LoRA weight initialization
    """
    def build(self, input_shape):
        self.conv.build(input_shape)
        if self.activate_lora and self.trainable:
            self.in_channels = int(input_shape[-1])
            self.lora.trainable = False
            flattened_kernel_dim = self.kernel_size * self.kernel_size * self.in_channels
            self.lora_a = self.add_weight(
                                shape=(self.kernel_size, self.kernel_size,self.in_channels, 4),
                                initializer="glorot_uniform",
                                trainable=True,
                                name="lora_a_kernel"
                            )
            
            self.lora_b = self.lora_B = self.add_weight(
                            shape=((self.kernel_size, self.kernel_size, 4, self.filters)),
                            initializer="zeros",
                            trainable=True,
                            name="lora_b_kernel"
                        )
            #self.conv_kernel = self.conv.kernel
            self.conv.trainable = False
            if self.normalized:
                self.batch_norm.momentum = 0.85
                self.batch_norm.trainable = True
        elif not self.activate_lora and self.trainable:
            self.lora.trainable = False
            self.conv.trainable = True
            self.batch_norm.trainable = True
        else:
            self.lora.trainable = False
            self.conv.trainable = False
            self.batch_norm.trainable = False
        super().build(input_shape)
        """

    # Build method with parallel adapter
    def build(self, input_shape):
        if self.activate_lora and self.trainable:
            self.lora.trainable = True
            self.conv.trainable = False
            if self.normalized:
                self.batch_norm.trainable = False
        elif not self.activate_lora and self.trainable:
            self.lora.trainable = False
            self.conv.trainable = True
            self.batch_norm.trainable = True
        else:
            self.lora.trainable = False
            self.conv.trainable = False
            self.batch_norm.trainable = False
        super().build(input_shape)

@register_keras_serializable(package="lBNReLuConvlora")
class ConvBlockLoRALayer(Layer):
    """
    A convolutional block: 1x1 convolution, 3x3 convolution, 1x1 convolution.
    """
    def __init__(
        self,
        output_filters: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        name: str = None,
        trainable: bool = True,
        activate_lora: bool = None,
        **kwargs
    ) -> None:
        super().__init__(name=name, trainable=trainable,**kwargs)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        self.activate_lora = activate_lora
        # Create layers

        self.bnrc1 = BatchNormReluConvLayerWLoRA(
        #self.bnrc1 = ConvBatchNormReluLayer(
            # 1x1 convolution
            filters=self.output_filters // 2,
            kernel_size=1,
            name="BNRC1",
            momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable if not self.activate_lora else False,
            use_relu=True,
            normalized = True,
            activate_lora = False #self.activate_lora
        )
        self.bnrc2 = BatchNormReluConvLayerWLoRA(
        #self.bnrc2 = ConvBatchNormReluLayer(
            # 3x3 convolution
            filters=self.output_filters // 2,
            kernel_size=3,
            name="BNRC2",
            momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable,
            use_relu=True,
            normalized = True,
            activate_lora = self.activate_lora
        )
        self.bnrc3 = BatchNormReluConvLayerWLoRA(
        #self.bnrc3 = ConvBatchNormReluLayer(
            # 1x1 convolution
            filters=self.output_filters,
            kernel_size=1,
            name="BNRC3",
            momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=trainable if not self.activate_lora else False,
            use_relu=True,
            normalized = True,
            activate_lora = False #self.activate_lora
        )
        
    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "activate_lora": self.activate_lora
            },
        }

    def call(self, inputs: tf.Tensor, training) -> tf.Tensor:
        x = self.bnrc1(inputs, training=training)
        x = self.bnrc2(x, training=training)
        x = self.bnrc3(x, training=training)
        return x
    
    def build(self, input_shape):
        super().build(input_shape)