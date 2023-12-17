# real_time_vst.py

"""Tools for real-time video style transfer, following Huang et al."""

__all__ = ["BaseStyledImageFactory", "load_image"]
__author__ = "joshuafajardo"
__version__ = "0.1.0"

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

from tqdm import tqdm


class RealTimeVstModel(tf.keras.Model):
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
        # Input: [style_image, video_frames, optical_flows]
        pass

class ResidualBlock(tf.keras.Layer):
    # Assumed shape of inputs: [batch_size, height, width, channels]
    def __init__(self):
        super().__init__()
        self.conv1 = ConvolutionalBlock(kernel_size=3, strides=1, channels=48, activation="relu")
        self.conv2 = ConvolutionalBlock(kernel_size=3, strides=1, channels=48, activation=None)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

class ConvolutionalBlock(tf.keras.Layer):
    # Assumed shape of inputs: [batch_size, height, width, channels]
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
    # Assumed shape of inputs: [batch_size, height, width, channels]
    def __init__(self, kernel_size, strides, channels, padding="same", activation=None):
        super().__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(channels, kernel_size, strides=strides, padding=padding)
        self.instance_norm = tf.keras.layers.Normalization(axis=(0, 3))
        self.activation = tf.keras.layers.Activation(activation)
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.instance_norm(x)
        x = self.activation(x)
