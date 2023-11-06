from util import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

CONTENT_IMAGE_PATH = "../../data/content/neckarfront.jpg"
STYLE_IMAGE_PATH = "../../data/style/starry-night.jpg"

content_image = load_image(CONTENT_IMAGE_PATH)
style_image = load_image(STYLE_IMAGE_PATH)

content_image = tf.image.resize(content_image, (768 // 2, 1024 // 2))
# style_image = tf.image.resize(style_image, (416 // 2, 525 // 2))
style_image = tf.image.resize(style_image, content_image.shape[:2])
factory = StyledImageFactory(content_image, style_image, learning_rate=7)
generated_image, losses = factory.generate_styled_image(num_epochs=30, clip_between_steps=False)
print(generated_image)
plt.imshow(tf.cast(generated_image, tf.int32))
plt.show()
plt.plot(np.arange(len(losses[0])), losses[0])
plt.show()
plt.plot(np.arange(len(losses[1])), losses[1])
plt.show()
plt.plot(np.arange(len(losses[2])), losses[2])
plt.show()
