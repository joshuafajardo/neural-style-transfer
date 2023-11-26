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
from math import ceil

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
        # We don't have enough memory to calculate the image_size x image_size
        # matrices (from Kernel calculations) directly. Therefore, we partition
        # these calculations.
        # There is room for improvement here (e.g. automated partitioning), but
        # we're okay with this simpler solution for now.
        NUM_PARTITIONS = 10

        _, map_height, map_width, num_maps = generated_maps.shape
        map_size = map_height * map_width
        simplified_shape = [map_size, num_maps]
        # Make tensors shape [N_l, M_l], like in the paper.
        # Ideally, this would be automatically done by the model.
        generated_maps = tf.transpose(
            tf.reshape(generated_maps, simplified_shape))
        target_maps = tf.transpose(
            tf.reshape(target_maps, simplified_shape))

        contribution = 0
        match self.kernel:
            case Kernel.LINEAR:
                def get_contribution(x, y, partition_num, partition_size):
                    """
                    TODO: Add docstring
                    """
                    y_start = partition_num * partition_size
                    y_end = (partition_num + 1) * partition_size
                    if y_start >= y.shape[1]:
                        return 0

                    kernel_calcs = tf.linalg.matmul(
                        x, y[:, y_start : y_end], transpose_a=True)
                    return tf.math.reduce_sum(kernel_calcs)
                    
                # The last partition may be smaller.
                partition_size = ceil(map_size / NUM_PARTITIONS)
                for i in range(NUM_PARTITIONS):
                    # From paper: k(f, f)
                    contribution += get_contribution(
                        generated_maps, generated_maps, i, partition_size)

                    # From paper: k(s, s)
                    contribution += get_contribution(
                        target_maps, target_maps, i, partition_size)

                    # From paper: -2k(f, s)
                    contribution += get_contribution(
                        generated_maps, target_maps, i, partition_size)

                factor = 1 / num_maps  # From paper: Z_k^l
                contribution *= factor
            case Kernel.POLY:
                factor = 1 / (num_maps ** 2)
                contribution *= factor
            case Kernel.GAUSSIAN:
                factor = 1
            case Kernel.BATCH_NORM:
                factor = 1 / num_maps
        print(contribution)
        return contribution

        

    def get_style_reps(self, feature_maps):
        # Since the MMD relies on the Kernel Trick, we don't explicitly
        # calculate the final style representations.
        raise Exception("Unexpected call to get_style_reps.")
