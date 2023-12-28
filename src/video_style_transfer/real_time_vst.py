# real_time_vst.py

"""Tools for real-time video style transfer, following Huang et al."""

__all__ = ["StylizingNetwork", "LossNetwork", "RealTimeVstFactory"]
__author__ = "joshuafajardo"
__version__ = "0.1.0"

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

from tqdm import tqdm

FLOAT_TYPE = tf.keras.backend.floatx()

class RealTimeVstFactory():
    def __init__(self, style_image, frames, flows, content_loss_weight=1,
                 style_loss_weight=10, temporal_weight=1e4,
                 total_variation_loss_weight=1e-3):
        """
        Initializes the model with a style image and a directory
        of content videos.

        style_image: tf.Tensor, shape [height, width, channels]
        frames: dict, where the keys are the video names and the values are
            tf.Tensors of shape [num_frames, height, width, channels]
        flows: dict, where the keys are the video names and the values are
            tf.Tensors of shape [num_frames - 1, height, width, 2]
        content_loss_weight: int/float
        style_loss_weight: int/float
        temporal_weight: int/float
        total_variation_loss_weight: int/float
        """
        self.loss_network = LossNetwork()
        self.stylizing_network = StylizingNetwork()
        style_image = tf.cast(tf.expand_dims(style_image, axis=0), FLOAT_TYPE)
        self.target_style_features = self.loss_network(style_image)["style_maps"]  # TODO Need to turn these into Gram matrices
        self.frames = self.cast_dict_values(frames, FLOAT_TYPE)
        self.flows = self.cast_dict_values(flows, FLOAT_TYPE)
        self.content_loss_weight = tf.cast(content_loss_weight, FLOAT_TYPE)
        self.style_loss_weight = tf.cast(style_loss_weight, FLOAT_TYPE)
        self.temporal_weight = tf.cast(temporal_weight, FLOAT_TYPE)
        self.total_variation_loss_weight = tf.cast(total_variation_loss_weight, FLOAT_TYPE)
    
    @staticmethod
    def cast_dict_values(dictionary, dtype):
        dictionary = dictionary.copy()
        for key in dictionary.keys():
            dictionary[key] = tf.cast(dictionary[key], dtype)
        return dictionary
    
    def train(self, epochs=2, learning_rate=1e-3):
        """Trains the model on the content videos."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Unlike in the paper, we count the number of epochs instead of the
        # number of iterations.
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for video_name in tqdm(self.frames.keys()):
                self.train_video(video_name, optimizer)
    
    def train_video(self, video_name, optimizer):
        """
        Trains the model on the frames of the video.

        In the paper, the authors train on a batch of 2 frames at a
        time. We constrain our training capabilities similarly.
        """
        frames = self.frames[video_name]
        flows = self.flows[video_name]
        num_frames = frames.shape[0]
        for frame_num in range(num_frames - 1):
            content_frames = frames[frame_num : frame_num + 2]
            flow = flows[frame_num]
            with tf.GradientTape() as tape:
                generated_frames = self.stylizing_network(content_frames)  # TODO verify that this works; we're relying on batch-ing
                spatial_losses = self.calc_spatial_losses(content_frames, generated_frames)
                temporal_loss = self.calc_temporal_loss(generated_frames, flow)
                total_loss = tf.reduce_sum(spatial_losses) + temporal_loss
            grads = tape.gradient(total_loss, self.stylizing_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.stylizing_network.trainable_variables))
    
    def calc_temporal_loss(self, generated_frames, flow):
        """
        Calculates the temporal loss between two generated frames,
        given the flow between their original content frames.
        
        generated_frames: tf.Tensor, shape [2, height, width, channels]
        flow: tf.Tensor, shape [height, width, 2]
        """
        return 0  # TODO


    def calc_spatial_losses(self, content_frames, generated_frames):
        """
        Calculates the spatial losses for the generated frames.

        content_frames: tf.Tensor, shape [batch_size, height, width, channels]
        generated_frames: tf.Tensor shape [batch_size, height, width, channels]
        """
        content_losses = self.content_loss_weight \
            * self.calc_content_losses(content_frames, generated_frames)  # TODO make sure that these are vectorized
        style_losses = self.style_loss_weight \
            * self.calc_style_losses(generated_frames)
        total_variation_losses = self.total_variation_loss_weight \
            * self.calc_total_variation_regularizers(generated_frames)

        return content_losses + style_losses + total_variation_losses

    def calc_content_losses(self, content_frames, generated_frames):
        """
        Calculates the content loss for each sample between the
        content and stylized frames, summed across all layers.

        content_frames: tf.Tensor, shape [batch_size, height, width, channels]
        generated_frames: tf.Tensor, shape [batch_size, height, width, channels]
        """
        target_features = self.loss_network(content_frames)["content_maps"]
        generated_features = self.loss_network(generated_frames)["content_maps"]
        total_losses = tf.zeros(generated_frames.shape[0])
        for layer in range(len(target_features)):
            layer_losses = tf.reduce_sum(
                tf.square(target_features[layer] - generated_features[layer]),  # TODO not sure if this is right
                axis=(1, 2, 3))
            divisor = tf.reduce_prod(tf.shape(target_features[layer])[1:])
            layer_losses = layer_losses / tf.cast(divisor, FLOAT_TYPE)
            total_losses = total_losses + layer_losses
        return total_losses

    def calc_style_losses(self, generated_frames):
        """
        Calculates the style loss for each generated sample, summed
        across all layers.
        
        generated_frames: tf.Tensor, shape [batch_size, height, width, channels]
        """
        generated_maps = self.loss_network(generated_frames)["style_maps"]
        generated_gram_matrices = self.calc_gram_matrices(generated_maps)  # TODO fix this. Most urgent, most likely to be wrong
        total_losses = tf.zeros(generated_frames.shape[0])
        for layer in range(len(self.target_style_features)):
            num_maps = generated_maps[layer].shape[-1]
            diffs = self.target_style_features[layer] - generated_gram_matrices[layer]  # TODO not sure if this is right
            layer_losses = tf.reduce_sum(tf.square(diffs), axis=(1, 2, 3))
            layer_losses = layer_losses / (num_maps ** 2)
            total_losses = total_losses + layer_losses
        return total_losses
    
    def calc_gram_matrices(self, feature_maps):
        """
        Calculates the Gram matrices of the feature maps for each sample.

        feature_maps: tf.Tensor, shape [batch_size, height, width, channels]
        """
        batch_size, height, width, channels = feature_maps.shape
        feature_maps = tf.reshape(feature_maps, (batch_size, height * width, channels))
        return tf.matmul(feature_maps, feature_maps, transpose_a=True)
    
    def calc_total_variation_regularizers(self, generated_frames):
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

        return tf.reduce_sum(tf.sqrt(horizontal_diffs + vertical_diffs),
                             axis=(1, 2, 3))


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
        self.deconv6 = DeconvolutionalBlock(kernel_size=3, strides=2, channels=32, activation="relu")
        self.deconv7 = DeconvolutionalBlock(kernel_size=3, strides=2, channels=16, activation="relu")
        self.conv8 = ConvolutionalBlock(kernel_size=3, strides=1, channels=3, activation="tanh")

    def call(self, inputs):
        # Input: [batch_size, height, width, channels]
        # Pixel values should be in [0, 255].
        print("original shape: ", inputs.shape)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.conv8(x)
        x = tf.clip_by_value(x, 0, 255)
        print("final shape: ", x.shape)
        return x

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvolutionalBlock(kernel_size=3, strides=1, channels=48, activation="relu")
        self.conv2 = ConvolutionalBlock(kernel_size=3, strides=1, channels=48, activation="linear")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

class ConvolutionalBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides, channels, padding="same", activation="linear"):
        # "Conv denotes the convolutional block (convolutional layer + instance
        # normalization + activation)"
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(channels, kernel_size, strides=strides, padding=padding)
        self.instance_norm = tf.keras.layers.Normalization(axis=(0, 3))  # TODO: Check
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.instance_norm(x)
        x = self.activation(x)
        return x

class DeconvolutionalBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides, channels, padding="same", activation="linear"):
        super().__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(channels, kernel_size, strides=strides, padding=padding)
        self.instance_norm = tf.keras.layers.Normalization(axis=(0, 3))
        self.activation = tf.keras.layers.Activation(activation)
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.instance_norm(x)
        x = self.activation(x)
        return x


def load_image(image_path):
    """
    Load the image found at the image path.
    """
    image = tf.keras.utils.load_img(image_path)
    return tf.keras.utils.img_to_array(image)