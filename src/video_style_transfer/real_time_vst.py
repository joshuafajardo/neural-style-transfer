# real_time_vst.py

"""Tools for real-time video style transfer, following Huang et al."""

__all__ = ["StylizingNetwork"]
__author__ = "joshuafajardo"
__version__ = "0.1.0"

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

from tqdm import tqdm

class RealTimeVstModel(tf.keras.Model):


class LossNetwork(tf.keras.Model):
    content_layers = ["block4_conv2"]
    style_layers = ["block1_conv2", "block2_conv2", "block3_conv2", "block4_conv2"]

    def __init__(self):
        super().__init__()
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights="imagenet")
        vgg.trainable = False
        
        # The paper doesn't mention replacing the max pooling layers with
        # average pooling layers, but we do it since it showed good results
        # with the original NST.
        vgg = replace_max_pooling_with_avg_pooling(vgg)

        content_maps = [vgg.get_layer(layer).get_output_at(-1)
                        for layer in self.content_layers]
        style_maps = [vgg.get_layer(layer).get_output_at(-1)
                      for layer in self.style_layers]
        outputs = {
            "content_maps": content_maps,
            "style_maps": style_maps
        }

        self.vgg_model = tf.keras.Model([vgg.input], outputs)

    def call(self, inputs):
        # Input: [batch_size, height, width, channels]
        # Pixel values should be in [0, 1].
        x = tf.keras.applications.vgg19.preprocess_input(inputs * 255)
        x = self.vgg_model(x)
        return x


# Ported from ../image_neural_style/transfer/base_nst.py
def replace_max_pooling_with_avg_pooling(model):
    """
    Creates a new model from an existing model, where the MaxPooling2D
    layers are replaced with AveragePooling2D layers.

    Average pooling allows for better gradient flow when optimizing
    the output image.
    """
    # Inspiration from https://stackoverflow.com/q/54508800
    prev_output = model.layers[0].output
    for i in range(1, len(model.layers)):
        original_layer = model.layers[i]
        if isinstance(original_layer, tf.keras.layers.MaxPooling2D):
            new_layer = tf.keras.layers.AveragePooling2D(
                pool_size=original_layer.pool_size,
                strides=original_layer.strides,
                padding=original_layer.padding,
                data_format=original_layer.data_format,
                name=original_layer.name,
            )
            prev_output = new_layer(prev_output)
        else:
            prev_output = original_layer(prev_output)
    return tf.keras.models.Model(inputs=model.input, outputs=prev_output)


class StylizingNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvolutionalBlock(kernel_size=3, strides=1, channels=16, activation="relu")
        self.conv2 = ConvolutionalBlock(kernel_size=3, strides=2, channels=32, activation="relu")
        self.conv3 = ConvolutionalBlock(kernel_size=3, strides=2, channels=48, activation="relu")
        self.res4 = ResidualBlock()
        self.res5 = ResidualBlock()
        self.deconv6 = DeconvolutionalBlock(kernel_size=3, strides=0.5, channels=32, activation="relu")
        self.deconv7 = DeconvolutionalBlock(kernel_size=3, strides=0.5, channels=16, activation="relu")
        self.conv8 = ConvolutionalBlock(kernel_size=3, strides=1, channels=3, activation="tanh")

    def call(self, inputs):
        # Input: [batch_size, height, width, channels]
        # Pixel values should be in [0, 1].
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.conv8(x)
        x = tf.clip_by_value(x, 0, 1)
        return x

class ResidualBlock(tf.keras.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvolutionalBlock(kernel_size=3, strides=1, channels=48, activation="relu")
        self.conv2 = ConvolutionalBlock(kernel_size=3, strides=1, channels=48, activation=None)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

class ConvolutionalBlock(tf.keras.Layer):
    def __init__(self, kernel_size, strides, channels, padding="same", activation=None):
        # "Conv denotes the convolutional block (convolutional layer + instance
        # normalization + activation)"
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(channels, kernel_size, strides=strides, padding=padding)
        self.instance_norm = tf.keras.layers.Normalization(axis=(0, 3))  # TODO: Check
        if activation != None:
            self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.instance_norm(x)
        x = self.activation(x)
        return x

class DeconvolutionalBlock(tf.keras.Layer):
    def __init__(self, kernel_size, strides, channels, padding="same", activation=None):
        super().__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(channels, kernel_size, strides=strides, padding=padding)
        self.instance_norm = tf.keras.layers.Normalization(axis=(0, 3))
        self.activation = tf.keras.layers.Activation(activation)
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.instance_norm(x)
        x = self.activation(x)