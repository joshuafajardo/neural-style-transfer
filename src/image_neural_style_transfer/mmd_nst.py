# domain_adaptation_nst.py

"""
Tools for understanding NST as a domain adaptation problem, following Li et al.
"""

__all__ = ["BaseStyledImageFactory", "load_image"]
__author__ = "joshuafajardo"
__version__ = "0.1.0"

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from tqdm import tqdm

from base_nst import BaseStyledImageFactory, load_image

class MMDStyledImageFactory(BaseStyledImageFactory):

    base_style_loss_weights = {  # Source: https://github.com/lyttonhao/Neural-Style-MMD/blob/master/neural-style.py
        "linear": 2e3,
        "poly": 1e-1,
        "gaussian": 3e14,
        "batch_norm": 1e3
    }

    def __init__(self, kernel, content_image, style_image,
                 balance_factor=1,
                 content_layers=BaseStyledImageFactory.default_content_layers,
                 style_layers=BaseStyledImageFactory.default_style_layers,
                 content_layer_weights=None,
                 style_layer_weights=None,
                 pooling="avg",
                 learning_rate=8):
        self.__setup_model(content_layers, style_layers, pooling)
        self.__set_kernel(kernel)

        self.output_shape = content_image.shape
        style_image = tf.image.resize(style_image, (self.output_shape[:2]))

        # Set the layer weights
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

        # Set the weights for the content and style losses.
        # The content loss weight is always set to 1 in the paper.
        self.content_loss_weight = 1
        self.style_loss_weight = \
            balance_factor * self.base_style_loss_weights[kernel]

        # Set the optimizer
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate)

        self.__set_targets(content_image, style_image)
    
    def __set_targets(self, content_image, style_image):
        """
        Sets the targets for the loss function, based on the content
        and style images.
        Unlike for the MMDStyledImageFactory, the style
        representations are not directly stored, since the style loss
        calculation relies on the Kernel trick.
        """
        content_model_out = self.model(self.preprocess(content_image))
        style_model_out = self.model(self.preprocess(style_image))

        content_maps = content_model_out["content_maps"]
        style_maps = style_model_out["style_maps"]

        self.target_content_reps = self.get_content_reps(content_maps)
        self.target_style_maps = style_maps

    def __set_kernel(self, kernel):
        match kernel:
            case "linear":
                self.kernel = self.linear_kernel
            case "poly":
                pass
            case "gaussian":
                pass
            case "batch_norm":
                pass

    def calc_style_loss(self, generated_style_maps):
        num_layers = len(generated_style_maps)

        contributions_list = []
        for layer in range(num_layers):
            curr_generated_maps = generated_style_maps[layer]
            (_, map_height, map_width, num_maps) = curr_generated_maps.shape
            map_size = map_height * map_width
            target = self.target_style_maps[layer]
            # TODO: Figure out how to calculate the factor, Z_k^l

        #     factor = 1 / (4 * (num_maps ** 2) * (map_size ** 2))
        #     contribution = factor * tf.math.reduce_sum(
        #         (generated_rep - target_rep) ** 2)
        #     contributions_list.append(contribution)
        # contributions_tensor = tf.stack(contributions_list)
        # return tf.tensordot(self.style_layer_weights, contributions_tensor, 1)
    
    def linear_kernel(x, y):
        pass
    

    def get_style_reps(self, feature_maps):
        # Since the MMD relies on the Kernel Trick, we don't explicitly
        # calculate the final style representations.
        raise Exception("Unexpected call to get_style_reps.")