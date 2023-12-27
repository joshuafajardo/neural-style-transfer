# real_time_vst.py

"""Tools for real-time video style transfer, following Huang et al."""

__all__ = ["StylizingNetwork", "LossNetwork", "RealTimeVstFactory"]
__author__ = "joshuafajardo"
__version__ = "0.1.0"

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

from tqdm import tqdm

class RealTimeVstFactory():
    def __init__(self, style_image, frames, flows, content_loss_weight=1,
                 style_loss_weight=1, total_variation_loss_weight=1):
        """
        Initializes the model with a style image and a directory
        of content videos.
        """
        self.loss_network = LossNetwork()
        self.stylizing_network = StylizingNetwork()
        self.target_style_features = self.loss_network(style_image)["style_maps"]
        self.frames = frames
        self.flows = flows
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight
        self.total_variation_loss_weight = total_variation_loss_weight
    
    def train_batch(self, content_videos, learning_rate=1e-3):
        """Trains the model on a batch of content videos."""
        pass

    def calc_spatial_loss(self, content_frame, generated_frame):
        """Calculates the spatial for the generated frame."""
        content_loss = self.content_loss_weight \
            * self.calc_content_loss(content_frame, generated_frame)
        style_loss = self.style_loss_weight \
            * self.calc_style_loss(generated_frame)
        total_variation_loss = self.total_variation_loss_weight \
            * self.calc_total_variation_regularizer(generated_frame)

    def calc_content_loss(self, content_frame, generated_frame):
        """
        Calculates the content loss between content and stylized
        frames summed across all layers.
        """
        target_features = self.loss_network(content_frame)["content_maps"]
        generated_features = self.loss_network(generated_frame)["content_maps"]
        total_loss = 0
        for layer in range(len(target_features)):
            layer_loss = tf.reduce_sum(
                tf.square(target_features[layer] - generated_features[layer]))
            layer_loss = layer_loss / tf.reduce_prod(
                tf.shape(target_features[layer])[1:])
            total_loss = total_loss + layer_loss
        return total_loss

    def calc_style_loss(self, generated_frames):
        generated_features = self.loss_network(generated_frames)["style_maps"]
        total_loss = 0
        for layer in range(len(self.target_style_features)):
            num_maps = generated_features[layer].shape[3]
            diffs = self.target_style_features[layer] - generated_features[layer]
            layer_loss = tf.reduce_sum(tf.square(diffs))
            layer_loss = layer_loss / (num_maps ** 2)
            total_loss = total_loss + layer_loss
        return total_loss
    
    def calc_total_variation_regularizer(self, generated_frames):
        """
        Calculates the total variation regularizer for the generated
        frames.
        """
        # We want the diff matrices to have the same shape.
        horizontal_diffs = generated_frames[:, :-1, 1:, :] \
            - generated_frames[:, :-1, :-1, :]
        vertical_diffs = generated_frames[:, 1:, :-1, :] \
            - generated_frames[:, :-1, :-1, :]

        horizontal_diffs = tf.square(horizontal_diffs)
        vertical_diffs = tf.square(vertical_diffs)

        return tf.reduce_sum(tf.sqrt(horizontal_diffs + vertical_diffs))

        


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


def load_image(image_path):
    """
    Load the image found at the image path.
    """
    image = tf.keras.utils.load_img(image_path)
    return tf.keras.utils.img_to_array(image)