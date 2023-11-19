# Neural Style Transfer

![neckarfront styled with starry night](generated/neckarfront-starry_night-avg_pooling.png)

This repository re-implements a variety of papers relating to Neural Style
Transfer (NST), introduced by Gatys et al. in "A Neural Algorithm of Artistic
Style".


## Projects Within This Repository
### [NST for Images](#nst-for-images)
- "A Neural Algorithm of Artistic Style" (Gatys et al.)
- Try it out in [Colab](https://colab.research.google.com/drive/1_vnwvTRRpNOcql8vib8MigMU7yOkI8VP?usp=sharing)!
- **Status:** Complete!

### Video Style Transfer
- "Real-Time Neural Style Transfer for Videos" (Huang et al.)
- **Status:** Implementation phase

### 3D Video Style Transfer
- "Stereoscopic Neural Style Transfer" (Chen et al.)
- **Status:** Reading phase


## NST For Images
This project is based on the paper that really kicked off the field of NST.
My implementation has two main goals:
1) **Stay as true as possible to the original implementation.**
2) **Organize the logic in a way that is easily comprehensible**.


### Fun Findings
**Average pooling >> Max Pooling**

The paper's authors recommend replacing VGG19's max pooling layers with average
pooling layers. However, in the initial phases of this project, I stuck to
using the VGG19 model as-is—I thought that the results would be fine.

The difference really is astounding, however.

![neckarfront-starry_night, max pooling](generated/neckarfront-starry_night-max_pooling.png)
![neckarfront-starry_night, average pooling](generated/neckarfront-starry_night-avg_pooling.png)

The result with max pooling is shown on the left, and the result with average
pooling is shown on the right. Disclaimer: the right image was generated with
a slightly higher weight for style loss in order to make the results more
comparable.

The colors in the right image really do pop out more than those in the left
image. The figures also look more stable, especially the tower in the back.