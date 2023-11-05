from util import *
import numpy as np
import matplotlib as plt

CONTENT_IMAGE_PATH = "../../data/content/neckarfront.jpg"
STYLE_IMAGE_PATH = "../../data/style/starry-night.jpg"

content_image = load_image(CONTENT_IMAGE_PATH)
style_image = load_image(STYLE_IMAGE_PATH)
factory = StyledImageFactory(content_image, style_image)
generated_image, losses = factory.generate_styled_image()
plt.imshow(generated_image)
plt.plot(np.arange(len(losses)), losses)
