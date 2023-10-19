import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib as plt


def transfer_style_to_content(
        content_image, style_image, initial_image=None, num_epochs=1000, learning_rate=0.01,
        alpha=10e-3, beta=1):
    model, preprocess, deprocess = setup_gatys_model()

    if initial_image is None:
        preprocessed_noise = preprocess(create_white_noise_image(content_image.shape))
        generated_image = tf.Variable(preprocessed_noise)
    else:
        generated_image = initial_image

    optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate)

    for i in range(num_epochs):
        with tf.GradientTape() as tape:
            loss = calc_total_loss(generated_image, content_image, style_image, alpha, beta)
        gradient = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([gradient, generated_image])

    return deprocess(generated_image)


def setup_gatys_model(pooling="avg"):
    model = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", pooling=pooling)
    model.trainable = False
    return model, tf.keras.applications.vgg19.preprocess_input, vgg19_deprocess


def calc_total_loss(generated_image, content_image, style_image, model, alpha, beta):
    pass


def calc_content_loss(generated_image, content_image, model):
    pass


def calc_style_loss(generated_image, style_image, model, layer_weights):
    pass


def calc_gram_matrix(feature_maps):
    pass


def load_image(image_path):
    pass


def create_white_noise_image(shape):
    """Create a uniformly random white noise image, with values in range [-128, 127) with the given shape."""
    return (np.random.rand(*shape)) * 255 - 128

def vgg19_deprocess(image):
    """Does the inverse of the VGG19 preprocessing step on the given image."""
    mean = [[[103.939, 116.779, 123.67]]]  # Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L61C9-L61C42
    image += mean
    return tfio.experimental.color.bgr_to_rgb(image)
