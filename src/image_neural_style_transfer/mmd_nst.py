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
            contribution = self.calc_normalized_mmd(
                self.kernel, curr_generated_maps, target_maps)
            contributions_list.append(contribution)
        contributions_tensor = tf.stack(contributions_list)
        return tf.tensordot(self.style_layer_weights, contributions_tensor, 1)
    
    @staticmethod
    # @tf.autograph.experimental.do_not_convert
    def calc_normalized_mmd(kernel, generated_maps, target_maps):
        _, map_height, map_width, num_maps = generated_maps.shape
        map_size = map_height * map_width
        simplified_shape = [map_size, num_maps]
        generated_maps = tf.reshape(generated_maps, simplified_shape)  # very little extra space used when reshaping
        target_maps = tf.reshape(target_maps, simplified_shape)

        contribution = 0
        if kernel == Kernel.LINEAR:
            def get_summed_kernel_vals(x, y):
                kernel_calcs = tf.linalg.matmul(
                    x, y, transpose_b=True)
                return tf.reduce_sum(kernel_calcs)

            contribution = contribution + get_summed_kernel_vals(
                generated_maps, generated_maps)
            contribution = contribution + get_summed_kernel_vals(
                target_maps, target_maps)
            contribution = contribution - 2 * get_summed_kernel_vals(
                generated_maps, target_maps)

            factor= 1 / (num_maps)
            contribution = contribution * factor
            return contribution
        elif kernel == Kernel.POLY:
            def get_summed_kernel_vals(x, y):
                kernel_calcs = tf.linalg.matmul(
                    x, y, transpose_b=True)
                return tf.reduce_sum(kernel_calcs ** 2)

            contribution = contribution + get_summed_kernel_vals(
                generated_maps, generated_maps)
            contribution = contribution + get_summed_kernel_vals(
                target_maps, target_maps)
            contribution = contribution - 2 * get_summed_kernel_vals(
                generated_maps, target_maps)

            factor= 1 / (num_maps ** 2)
            contribution = contribution * factor
            return contribution
        elif kernel == Kernel.GAUSSIAN:
            def get_unbiased_mmd_estimate(x, y):
                def sample_pairs_without_replacement(cardinality):
                    rng = np.random.default_rng()
                    flattened_samples = rng.choice(
                        cardinality ** 2, size=cardinality, replace=False)
                    # x indices, y indices
                    return [flattened_samples // cardinality,
                            flattened_samples % cardinality]
                def get_squared_l2_norms(vectors):
                    # Assumes vectors[i] is the ith vector.
                    return tf.reduce_sum(tf.math.pow(vectors, 2), axis=1)
                    
                num_samples = x.shape[0] & ~1  # Want even num_samples
                indices = sample_pairs_without_replacement(num_samples)

                x_samples = tf.gather(x, indices[0])
                y_samples = tf.gather(y, indices[1])

                x_even = x_samples[0::2]
                x_odd = x_samples[1::2]
                y_even = y_samples[0::2]
                y_odd = y_samples[1::2]

                diffs = {
                    "xx": x_odd - x_even,
                    "yy": y_odd - y_even,
                    "xy": x_odd - y_even,
                    "yx": y_odd - x_even,
                }

                squared_norms = {
                    "xx": get_squared_l2_norms(diffs["xx"]),
                    "yy": get_squared_l2_norms(diffs["yy"]),
                    "xy": get_squared_l2_norms(diffs["xy"]),
                    "yx": get_squared_l2_norms(diffs["yx"]),
                }

                gamma = num_samples / tf.reduce_sum(list(squared_norms.values()))

                kernel_outs = {
                    "xx": tf.math.exp(-gamma * squared_norms["xx"]),
                    "yy": tf.math.exp(-gamma * squared_norms["yy"]),
                    "xy": tf.math.exp(-gamma * squared_norms["xy"]),
                    "yx": tf.math.exp(-gamma * squared_norms["yx"]),
                }

                return tf.reduce_sum(kernel_outs["xx"] + kernel_outs["yy"]
                    - kernel_outs["xy"] - kernel_outs["yx"]) / num_samples
            return get_unbiased_mmd_estimate(generated_maps, target_maps)

        # case Kernel.BATCH_NORM:
        #     factor = 1 / num_maps

    def get_style_reps(self, feature_maps):
        # Since the MMD relies on the Kernel Trick, we don't explicitly
        # calculate the final style representations.
        raise Exception("Unexpected call to get_style_reps.")
