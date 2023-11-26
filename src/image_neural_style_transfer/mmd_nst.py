# domain_adaptation_nst.py

"""
Tools for understanding NST as a domain adaptation problem, following Li et al.
"""

__all__ = ["Kernel", "MMDStyledImageFactory", "load_image"]
__author__ = "joshuafajardo"
__version__ = "0.1.0"

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from tqdm import tqdm
from enum import Enum

from base_nst import BaseStyledImageFactory, load_image

class Kernel(Enum):
    LINEAR = 1
    POLY = 2
    GAUSSIAN = 3
    BATCH_NORM = 4

class MMDStyledImageFactory(BaseStyledImageFactory):

    standard_style_loss_weights = {  # Source: https://github.com/lyttonhao/Neural-Style-MMD/blob/master/neural-style.py
        Kernel.LINEAR: 2e3,
        Kernel.POLY: 1e-1,
        Kernel.GAUSSIAN: 3e14,
        Kernel.BATCH_NORM: 1e3
    }

    def __init__(self, kernel, content_image, style_image,
                 balance_factor=1,
                 content_layers=BaseStyledImageFactory.default_content_layers,
                 style_layers=BaseStyledImageFactory.default_style_layers,
                 content_layer_weights=None,
                 style_layer_weights=None,
                 pooling="avg",
                 learning_rate=8):
        self.setup_model(content_layers, style_layers, pooling)
        self.kernel = kernel

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
            balance_factor * self.standard_style_loss_weights[kernel]

        # Set the optimizer
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate)

        self.set_targets(content_image, style_image)
    
    def set_targets(self, content_image, style_image):
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

    def calc_style_loss(self, generated_style_maps):
        num_layers = len(generated_style_maps)

        contributions_list = []
        for layer in range(num_layers):
            curr_generated_maps = generated_style_maps[layer]
            target_maps = self.target_style_maps[layer]
            contribution = self.calc_normalized_mmd(curr_generated_maps, target_maps)
            contributions_list.append(contribution)
        contributions_tensor = tf.stack(contributions_list)
        return tf.tensordot(self.style_layer_weights, contributions_tensor, 1)
    
    def calc_normalized_mmd(self, generated_maps, target_maps):
        num_maps = generated_maps.shape[3]
        contribution = 0
        dot_axes = [[3], [3]]
        match self.kernel:
            case Kernel.LINEAR:
                generated_generated_matrix = tf.tensordot(
                    generated_maps, generated_maps, axes=dot_axes)
                contribution += tf.math.reduce_sum(generated_generated_matrix)

                target_target_matrix = tf.tensordot(
                    target_target_matrix, target_target_matrix, axes=dot_axes)
                contribution += tf.math.reduce_sum(target_target_matrix)

                generated_target_matrix = tf.tensordot(
                    generated_maps, target_maps, axes=dot_axes)
                contribution -= 2 * tf.math.reduce_sum(generated_target_matrix)

                factor = 1 / num_maps
                contribution *= factor
            case Kernel.POLY:
                generated_generated_matrix = tf.tensordot(
                    generated_maps, generated_maps, axes=dot_axes)
                contribution += tf.math.reduce_sum(generated_generated_matrix ** 2)

                target_target_matrix = tf.tensordot(
                    target_target_matrix, target_target_matrix, axes=dot_axes)
                contribution += tf.math.reduce_sum(target_target_matrix ** 2)

                generated_target_matrix = tf.tensordot(
                    generated_maps, target_maps, axes=dot_axes)
                contribution -= 2 * tf.math.reduce_sum(generated_target_matrix ** 2)

                factor = 1 / (num_maps ** 2)
                contribution *= factor
            case Kernel.GAUSSIAN:
                factor = 1
            case Kernel.BATCH_NORM:
                factor = 1 / num_maps
        return contribution

        

    def get_style_reps(self, feature_maps):
        # Since the MMD relies on the Kernel Trick, we don't explicitly
        # calculate the final style representations.
        raise Exception("Unexpected call to get_style_reps.")
