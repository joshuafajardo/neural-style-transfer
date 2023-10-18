import tensorflow as tf
import numpy as np
import matplotlib as plt


def transfer_style_to_content(content_image, style_image, model):
    pass

def setup_model():
    pass

def setup_gatys_model(pooling="avg"):
    model = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", pooling=pooling)
    model.trainable = False
    return model, tf.keras.applications.vgg19.preprocess_input

def calc_gram_matrix(X):
    pass

def load_image(image_path):
    pass
