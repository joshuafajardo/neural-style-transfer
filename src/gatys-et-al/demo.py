from util import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

CONTENT_IMAGE_PATH = "../../data/content/neckarfront.jpg"
STYLE_IMAGE_PATH = "../../data/style/starry-night.jpg"

with tf.device("/GPU:0"):
    content_image = load_image(CONTENT_IMAGE_PATH)
    style_image = load_image(STYLE_IMAGE_PATH)
    content_image = tf.image.resize(content_image, (768 // 3, 1024 // 3))

    factory = StyledImageFactory(content_image, style_image)
    generated_image, losses = factory.generate_styled_image(num_epochs=3000, clip_between_steps=True)

print(generated_image)
plt.imshow(tf.cast(generated_image, tf.int32))
plt.show()