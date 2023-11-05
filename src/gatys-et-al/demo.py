from util import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

CONTENT_IMAGE_PATH = "../../data/content/neckarfront.jpg"
STYLE_IMAGE_PATH = "../../data/style/starry-night.jpg"

content_image = load_image(CONTENT_IMAGE_PATH)
style_image = load_image(STYLE_IMAGE_PATH)

content_image = tf.image.resize(content_image, (768 // 2, 1024 // 2))
style_image = tf.image.resize(style_image, (416 // 2, 525 // 2))
factory = StyledImageFactory(content_image, style_image)
generated_image, losses = factory.generate_styled_image(num_epochs=50)
print(generated_image)
plt.imshow(generated_image)
plt.show()
plt.plot(np.arange(len(losses)), losses)
plt.show()
