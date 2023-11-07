# util.py

"""Provides tools for transfering style, following Gatys et al."""

__all__ = ["StyledImageFactory", "load_image"]
__author__ = "joshuafajardo"
__version__ = "0.1.0"

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


IMAGENET_MEAN = [103.939, 116.779, 123.67]  # Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L61C9-L61C42
FLOAT_TYPE = tf.keras.backend.floatx()

class StyledImageFactory():
    DEFAULT_CONTENT_LAYERS = ['block4_conv2'] 
    DEFAULT_STYLE_LAYERS = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    def __init__(self, content_image, style_image,
                 content_layers=DEFAULT_CONTENT_LAYERS,
                 style_layers=DEFAULT_STYLE_LAYERS,
                 content_layer_weights=None,
                 style_layer_weights=None,
                 content_loss_weight=10e-4,
                 style_loss_weight=1,
                 learning_rate=0.001):
        """Initialize the StyledImageFactory."""
        self.__setup_model(content_layers, style_layers)

        self.output_shape = content_image.shape
        style_image = tf.image.resize(style_image, (self.output_shape[:2]))

        if content_layer_weights is None:
            num_content_layers = len(content_layers)
            content_layer_weights = [1 / num_content_layers] \
                * num_content_layers
        self.content_layer_weights = tf.convert_to_tensor(
            content_layer_weights)
        if style_layer_weights is None:
            num_style_layers = len(style_layers)
            style_layer_weights = [1 / num_style_layers] * num_style_layers
        self.style_layer_weights = tf.convert_to_tensor(style_layer_weights)

        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight

        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate)

        # Set up the target representations.
        content_model_out = self.model(self.preprocess(content_image))
        style_model_out = self.model(self.preprocess(style_image))

        content_maps = content_model_out["content_maps"]
        style_maps = style_model_out["style_maps"]

        self.target_content_reps = self.get_content_reps(content_maps)
        self.target_style_reps = self.get_style_reps(style_maps)

    def __setup_model(self, content_layers, style_layers):
        """
        Sets up an internal "model", where the input is the image to be
        improved, and the output contains the activations (aka
        representations) of the specified content and style layers.
        """
        vgg_model = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights="imagenet", pooling="avg")
        vgg_model.trainable = False

        # "maps" as in "feature maps"
        content_maps = [vgg_model.get_layer(layer).output for layer in content_layers]
        style_maps = [vgg_model.get_layer(layer).output for layer in style_layers]
        outputs = {
            "content_maps": content_maps,
            "style_maps": style_maps
        }

        self.model = tf.keras.Model([vgg_model.input], outputs)
    
    def generate_styled_image(self, initial_image=None, num_epochs=1000,
                              clip_between_steps=False):
        """
        Note: While the original paper doesn't mention anything about
        clipping the image between optimizer steps, we do it in order
        to improve the optimization.
        """  # TODO: Improve the wording
        # Initialize the input to the model.
        if initial_image is None:
            initial_image = self.create_white_noise_image(
                self.output_shape)
        preprocessed_image = self.preprocess(initial_image)
        generated_image = tf.Variable(preprocessed_image)

        losses = self.calc_losses(
            generated_image)
        losses_across_epochs = [losses]

        # Optimize the image.
        for _ in tqdm(range(num_epochs)):
            losses = self.run_optimizer_step(
                generated_image)
            losses_across_epochs.append(losses)

            if clip_between_steps:
                clipped = self.clip_to_valid_range(generated_image)
                generated_image.assign(clipped)

        return self.deprocess(generated_image), losses
    
    @staticmethod
    @tf.function()
    def clip_to_valid_range(image):
        mean = tf.reshape(IMAGENET_MEAN, (1, 1, 1, 3))
        mean = tf.cast(mean, image.dtype)
        lower_bound = 0 - mean
        upper_bound = 255 - mean
        image = tf.maximum(image, lower_bound)
        image = tf.minimum(image, upper_bound)
        return image

    @tf.function()
    def run_optimizer_step(self, image):
        with tf.GradientTape() as tape:
            losses = self.calc_losses(image)
        self.optimizer.minimize(losses["total"], [image], tape=tape)
        return losses

    @tf.function()
    def calc_losses(self, image):
        losses = {}
        model_output = self.model(image)
        losses["content"] = self.calc_content_loss(model_output["content_maps"])
        losses["style"] = self.calc_style_loss(model_output["style_maps"])
        losses["total"] = (self.content_loss_weight * losses["content"]) \
            + (self.style_loss_weight * losses["style"]) 
        # Not necessary to return all 3, but helps with debugging and graphing.
        return losses

    @tf.function()
    def calc_content_loss(self, generated_content_maps):
        num_layers = len(generated_content_maps)
        generated_reps = self.get_content_reps(generated_content_maps)

        contributions_list = []
        for layer in range(num_layers):
            generated_rep = generated_reps[layer]
            target_rep = self.target_content_reps[layer]
            
            contribution = 0.5 * tf.math.reduce_sum(
                (generated_rep - target_rep) ** 2)
            contributions_list.append(contribution)
        contributions_tensor = tf.stack(contributions_list)
        return tf.tensordot(self.content_layer_weights,
                            contributions_tensor, 1)

    @tf.function()
    def calc_style_loss(self, generated_style_maps):
        num_layers = len(generated_style_maps)
        generated_reps = self.get_style_reps(generated_style_maps)

        contributions_list = []
        for layer in range(num_layers):
            generated_map = generated_style_maps[layer]
            (_, map_height, map_width, num_maps) = generated_map.shape
            map_size = map_height * map_width

            generated_rep = generated_reps[layer]
            target_rep = self.target_style_reps[layer]

            factor = 1 / (4 * (num_maps ** 2) * (map_size ** 2))
            contribution = factor * tf.math.reduce_sum(
                (generated_rep - target_rep) ** 2)
            contributions_list.append(contribution)
        contributions_tensor = tf.stack(contributions_list)
        return tf.tensordot(self.style_layer_weights, contributions_tensor, 1)

    @tf.function()
    def get_content_reps(self, feature_maps):
        return feature_maps

    @tf.function()
    def get_style_reps(self, feature_maps):
        reps = []
        for map in feature_maps:
            reps.append(self.calc_gram_matrix(map))
        return reps
        
    @staticmethod
    @tf.function()
    def calc_gram_matrix(feature_map):
        # In the paper, the feature map for layer l has shape (N_l, M_l), where
        # N_l is the number of feature maps, and M_l is the height * width of
        # the map.
        # Our actual feature maps have shape (1, height, width, N_l).
        # Therefore, the steps we take here are slightly different.
        feature_map_T = tf.transpose(feature_map, [0, 3, 1, 2])
        gram_matrix = tf.tensordot(
            feature_map_T, feature_map, [[2, 3], [1, 2]])
        return tf.squeeze(gram_matrix, [0, 2])

    @staticmethod
    def create_white_noise_image(shape):
        """
        Create a uniformly random white noise image, with values in range
        [0, 255) and with the given shape.
        """
        return np.random.rand(*shape) * 255 
    
    @staticmethod
    def preprocess(image):
        image = tf.keras.applications.vgg19.preprocess_input(image)
        return tf.expand_dims(image, axis=0)
    
    @staticmethod
    def deprocess(image):
        image = tf.squeeze(image, [0])
        mean = tf.reshape(IMAGENET_MEAN, (1, 1, 3))
        mean = tf.cast(mean, image.dtype)
        image = image + mean
        image = tf.clip_by_value(image, 0, 255)
        return tfio.experimental.color.bgr_to_rgb(image)

def load_image(image_path):
    image = tf.keras.utils.load_img(image_path)
    return tf.keras.utils.img_to_array(image)
