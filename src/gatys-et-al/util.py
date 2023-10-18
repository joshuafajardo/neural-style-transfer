import tensorflow as tf
import numpy as np
import matplotlib as plt


def transfer_style_to_content(content_image, style_image, model, preprocess, initial_image=None):
    if initial_image == None:
        preprocessed_noise = preprocess(create_white_noise_image(content_image.shape))
        initial_image = tf.Variable(preprocessed_noise)
    



def setup_model():
    pass


def setup_gatys_model(pooling="avg"):
    model = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", pooling=pooling)
    model.trainable = False
    return model, tf.keras.applications.vgg19.preprocess_input


def calc_total_loss(generated_image, content_image, style_image, alpha, beta):
    pass


def calc_content_loss(generated_image, content_image):
    pass


def calc_style_loss(generated_image, style_image, layer_weights):
    pass


def calc_gram_matrix(feature_maps):
    pass


def load_image(image_path):
    pass


def create_white_noise_image(shape):
    return np.random.randn(*shape)
