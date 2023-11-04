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
                 content_loss_weight=10e-3, style_loss_weight=1):
        self.content_image = content_image
        self.style_image = style_image
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight
        self.setup_model(content_layers, style_layers)

    def generate_styled_image(self, initial_image=None, num_epochs=1000,
                              learning_rate=0.01):
        # Initialize the input to the model.
        if initial_image is None:
            initial_image = self.create_white_noise_image(
                self.content_image.shape)
        preprocessed_image = self.preprocess(initial_image)
        generated_image = tf.Variable(preprocessed_image)

        optimizer = tf.keras.optimizers.experimental.SGD(
            learning_rate=learning_rate)
        # TODO: run gradient descent on the image.
        return self.deprocess_image(generated_image)

    def setup_model(self, content_layers, style_layers):
        """
        Sets up an internal "model", where the input is the image to be
        improved, and the output contains the activations (aka
        representations) of the specified content and style layers.
        """
        vgg_model = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights="imagenet", pooling="avg")
        vgg_model.trainable = False

        # "reps" as in "representations"
        content_reps = [vgg_model.get_layer(layer) for layer in content_layers]
        style_reps = [vgg_model.get_layer(layer) for layer in style_layers]
        outputs = {
            "content_reps": content_reps,
            "style_reps": style_reps
        }

        self.model = tf.keras.Model([vgg_model.input], outputs)

    def calc_total_loss():
        pass

    def calc_content_loss():
        pass

    def calc_style_loss():
        pass

    def calc_gram_matrix(feature_maps):
        pass

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
        return tfio.experimental.color.bgr_to_rgb(image)

def load_image(image_path):
    return tf.keras.utils.load_img(image_path)
