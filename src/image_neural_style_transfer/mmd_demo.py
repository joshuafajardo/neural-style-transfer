from mmd_nst import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

CONTENT_IMAGE_PATH = "../../data/content/neckarfront.jpg"
STYLE_IMAGE_PATH = "../../data/style/starry-night.jpg"

with tf.device("/GPU:0"):
    content_image = load_image(CONTENT_IMAGE_PATH)
    style_image = load_image(STYLE_IMAGE_PATH)
    content_image = tf.image.resize(content_image, (768 // 8, 1024 // 8))

    factory = MMDStyledImageFactory(Kernel.POLY, content_image, style_image, balance_factor=10e10)
    generated_image, losses = factory.generate_styled_image(num_epochs=1)

plt.imshow(tf.cast(generated_image, tf.int32))
plt.show()
