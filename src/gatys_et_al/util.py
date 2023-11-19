# util.py

"""Provides tools for transfering style, following Gatys et al."""

__all__ = ["StyledImageFactory", "load_image"]
__author__ = "joshuafajardo"
__version__ = "0.1.0"

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

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
                 pooling="avg",
                 learning_rate=8):
        """Initialize the StyledImageFactory."""
        self.__setup_model(content_layers, style_layers, pooling)

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

        # Set the weights for content, style loss
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight

        # Set the optimizer
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

    def __setup_model(self, content_layers, style_layers, pooling):
        """
        Sets up an internal "model", where the input is the image to be
        improved, and the output contains the activations (aka
        representations) of the specified content and style layers.
        """
        vgg_model = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights="imagenet")
        vgg_model.trainable = False

        if pooling == "avg":
            vgg_model = self.replace_max_pooling_with_avg_pooling(vgg_model)

        # "maps" as in "feature maps"
        content_maps = [vgg_model.get_layer(layer).get_output_at(-1) for layer in content_layers]
        style_maps = [vgg_model.get_layer(layer).get_output_at(-1) for layer in style_layers]
        outputs = {
            "content_maps": content_maps,
            "style_maps": style_maps
        }

        self.model = tf.keras.Model([vgg_model.input], outputs)
    
    @staticmethod
    def replace_max_pooling_with_avg_pooling(model):
        """
        Creates a new model from the existing model, where the MaxPooling2D
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
        
    
    def generate_styled_image(self, initial_image=None, num_epochs=3000,
                              clip_between_steps=True):
        """
        Generates a new image, based on the content and style sources that were
        provided when constructing the factory.
        """
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
    def create_white_noise_image(shape):
        """
        Create a random white noise image with the given shape.
        """
        # Normal distribution; most values within [0-255] (~6 sigma).
        image = np.random.normal(loc=127, scale=45, size=shape)
        return tf.cast(image, FLOAT_TYPE)
    
    @staticmethod
    def clip_to_valid_range(image):
        """
        Clips the intermediate image to the valid range of values.
        """
        mean = tf.reshape(IMAGENET_MEAN, (1, 1, 1, 3))
        mean = tf.cast(mean, image.dtype)
        lower_bound = 0 - mean
        upper_bound = 255 - mean
        image = tf.maximum(image, lower_bound)
        image = tf.minimum(image, upper_bound)
        return image

    @tf.function(reduce_retracing=True)
    def run_optimizer_step(self, image):
        """
        Run one optimization step on the image.
        """
        with tf.GradientTape() as tape:
            losses = self.calc_losses(image)
        self.optimizer.minimize(losses["total"], [image], tape=tape)
        return losses

    def calc_losses(self, image):
        """
        Calculate all losses (total, content, and style) for the given
        image.
        """
        losses = {}
        model_output = self.model(image)
        losses["content"] = self.calc_content_loss(model_output["content_maps"])
        losses["style"] = self.calc_style_loss(model_output["style_maps"])
        losses["total"] = (self.content_loss_weight * losses["content"]) \
            + (self.style_loss_weight * losses["style"]) 
        # Not necessary to return all 3, but helps with debugging and graphing.
        return losses

    def calc_content_loss(self, generated_content_maps):
        """
        Calculate the content loss, given the content maps generated by
        the model.
        """
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

    def calc_style_loss(self, generated_style_maps):
        """
        Calculate the style loss, given the style maps generated by the
        model.
        """
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

    def get_content_reps(self, feature_maps):
        """
        Get the content representation from the content feature maps.
        """
        return feature_maps

    def get_style_reps(self, feature_maps):
        """
        Get the style representation from the style feature maps.
        """
        reps = []
        for map in feature_maps:
            reps.append(self.calc_gram_matrix(map))
        return reps
        
    @staticmethod
    def calc_gram_matrix(feature_map):
        """
        Calculate the gram matrix for the given feature map.
        """
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
    def preprocess(image):
        """
        Prepare the image to be processed by the factory.
        """
        image = tf.keras.applications.vgg19.preprocess_input(image)
        return tf.expand_dims(image, axis=0)
    
    @staticmethod
    def deprocess(image):
        """
        Deprocess the generated intermediate image.
        """
        image = tf.squeeze(image, [0])
        mean = tf.reshape(IMAGENET_MEAN, (1, 1, 3))
        mean = tf.cast(mean, image.dtype)
        image = image + mean
        image = tf.clip_by_value(image, 0, 255)
        return tfio.experimental.color.bgr_to_rgb(image)

def load_image(image_path):
    """
    Load the image found at the image path.
    """
    image = tf.keras.utils.load_img(image_path)
    return tf.keras.utils.img_to_array(image)
