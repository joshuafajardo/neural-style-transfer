# util.py

"""Provides tools for transfering style, following Gatys et al."""

__all__ = ["StyledImageFactory", "load_image"]
__author__ = "joshuafajardo"
__version__ = "0.1.0"

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib as plt


IMAGENET_MEAN = [[[103.939, 116.779, 123.67]]]  # Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L61C9-L61C42

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
                 content_loss_weight=10e-3,
                 style_loss_weight=1,
                 learning_rate=0.01):
        """Initialize the StyledImageFactory."""
        self.__setup_model(content_layers, style_layers)

        if content_layer_weights is None:
            content_layer_weights = [1] * len(content_layers)
        self.content_layer_weights = tf.convert_to_tensor(content_layer_weights)
        if style_layer_weights is None:
            style_layer_weights = [1] * len(style_layers)
        self.style_layer_weights = tf.convert_to_tensor(style_layer_weights)

        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight

        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.experimental.SGD(
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
        content_maps = [vgg_model.get_layer(layer) for layer in content_layers]
        style_maps = [vgg_model.get_layer(layer) for layer in style_layers]
        outputs = {
            "content_maps": content_maps,
            "style_maps": style_maps
        }

        self.model = tf.keras.Model([vgg_model.input], outputs)
    
    def generate_styled_image(self, initial_image=None, num_epochs=1000,
                              clip_between_steps=True):
        """
        Note: While the original paper doesn't mention anything about
        clipping the image between optimizer steps, we do it in order
        to improve the optimization.
        """  # TODO: Improve the wording
        # Initialize the input to the model.
        if initial_image is None:
            initial_image = self.create_white_noise_image(
                self.content_image.shape)
        preprocessed_image = self.preprocess(initial_image)
        generated_image = tf.Variable(preprocessed_image)
        losses = [self.calc_total_loss(generated_image)]

        for epoch in range(num_epochs):
            loss = self.run_optimizer_step(generated_image)
            losses.append(loss)

        return self.deprocess_image(generated_image), losses

    @tf.function()
    def run_optimizer_step(self, image):
        with tf.GradientTape() as tape:
            loss = self.calc_total_loss(image)
        self.optimizer.minimize(loss, [image], tape=tape)
        return loss

    @tf.function()
    def calc_total_loss(self, image):
        model_output = self.model(image)
        content_loss = self.calc_content_loss(model_output["content_maps"])
        style_loss = self.calc_style_loss(model_output["style_maps"])
        return (self.content_loss_weight * content_loss) \
            + (self.style_loss_weight * style_loss)

    @tf.function()
    def calc_content_loss(self, generated_content_reps):
        num_layers = len(generated_content_reps)
        contributions = tf.zeros(num_layers)
        for layer in num_layers:
            generated_rep = generated_content_reps[layer]
            target_rep = self.target_content_reps[layer]
            
            contributions[layer] = 0.5 * tf.math.reduce_sum(
                (generated_rep - target_rep) ** 2)
        return tf.tensordot(self.content_layer_weights, contributions)

    @tf.function()
    def calc_style_loss(self, generated_style_reps):
        num_layers = len(generated_style_reps)
        contributions = tf.zeros(num_layers)
        for layer in num_layers:
            generated_rep = generated_style_reps[layer]
            target_rep = self.target_style_reps[layer]
            num_maps, map_size = generated_rep.shape

            factor = 1 / (4 * (num_maps ** 2) * (map_size ** 2))
            contributions[layer] = factor * tf.math.reduce_sum(
                (generated_rep - target_rep) ** 2)
        return tf.tensordot(self.style_layer_weights, contributions)

    @tf.function()
    def get_content_reps(self, feature_maps):
        """
        Assumes that self.model is already set.
        """
        return feature_maps

    @tf.function()
    def get_style_reps(self, feature_maps):
        """
        Assumes that self.model is already set.
        """
        reps = []
        for map in feature_maps:
            reps.append(self.calc_gram_matrix(map))
        return reps
        

    @tf.function()
    def calc_gram_matrix(feature_map):
        return tf.matmul(feature_map, feature_map,
                         transpose_b=True)

    def create_white_noise_image(shape):
        """
        Create a uniformly random white noise image, with values in range
        [-128, 127) and with the given shape.
        """
        return (np.random.rand(*shape)) * 255 - 128
    
    def preprocess(image):
        return tf.keras.applications.vgg19.preprocess_input(image)
    
    def deprocess(image):
        image += IMAGENET_MEAN
        image = np.clip(image, 0, 255)
        return tfio.experimental.color.bgr_to_rgb(image)

def load_image(image_path):
    return tf.keras.utils.load_img(image_path)
