# Gan-getting-started
A kaggle competition - ["I'm Something of a Painter Myself"](https://www.kaggle.com/competitions/gan-getting-started)

# Week 5: GANs

The purpose of this notebook is to use a Generative Neural Network architecture to generate images that appear similar in style to paintings by famous French impressionist painter Claude Monet. The project is for a Kaggle competition titled "I'm Something of a Painter Myself". The data, provided by Kaggle, consists of 256x256 pixel RGB images. There are 300 Monet paintings and 7038 real photographs. Using these we trained GANs to learn features of Monet's distinct style.

In this notebook, two separate attempts using two different types of GAN were made towards this goal. The first involved a DCGAN with an attempt to generate Monet style pictures from random latent vectors. The second attempt uses a CycleGAN to "Monet-ify" the photos in our dataset.

## Import

First we import the data and set some hyperparameters that we will need throughout the notebook.

## Data Inspection and Exploratory Analysis

Next, we investigate the files structure. What features about the image are included in the Kaggle dataset? It appears each photo has 3 features included, the target, image name, and the image data.

First, we check the size of our two targets. It appears that there are far more (7,038) photos than Monet paintings (300). The size of the Monet dataset seems rather small which could pose a challenge for our networks to learn essential features.

![image](https://github.com/user-attachments/assets/397e09f5-831d-4b50-85c9-d52b31f783e0)

We visualized the first batch of Monet images and photos.

![image](https://github.com/user-attachments/assets/1ec4de76-ef01-473d-97a6-d4a86b1bd7b2)

![image](https://github.com/user-attachments/assets/0bfb32d7-6fe7-403b-99e4-c3e74e0f6028)

Visualize the RGB Channels for each dataset. From this analysis it appears that the Monet paintings tend use deeper more intense colors. This tendency appears most prominently in the Green channel, but exists across all three channels.

![image](https://github.com/user-attachments/assets/6ce5b4cc-6b0f-40e2-bd9d-a5683f1dfa2e)

## Model DCGAN (attempt 1)

The first GAN architecture we employed was a DCGAN. Like all GANs and like the term 'adversarial' implies, the DCGAN consists of two competing actors, a Discriminator that tries to classify real and fake images (in this case Monet and not-Monet), and a Generator that attempts to create images that can fool the discriminator. Each one is constantly learning by updating its parameters to improve its performance at its task. 

The DCGAN architecture builds on the fundamental adversarial process of GANs, where a generator and discriminator engage in a competitive learning process to produce increasingly realistic images. The generator starts from a latent vector, progressively transforming it into a high-resolution 256×256 image through a sequence of transposed convolutions, each followed by batch normalization and LeakyReLU activation. This layered approach ensures both fine-grained detail retention and controlled gradient flow, preventing instability during training. The final activation function, tanh, scales the generated output to match the range of real images.

The discriminator, in contrast, is a fully convolutional classifier designed to distinguish authentic Monet paintings from synthetic ones. Using a series of convolutional layers, dropout, and LeakyReLU, it reduces the image spatial dimensions while capturing essential structural patterns. Rather than applying a sigmoid activation at the output, it directly returns logits, allowing for binary cross-entropy loss with soft labels, which improves generalization and adversarial robustness.

Training is guided by an adversarial loss function where the generator strives to fool the discriminator into misclassifying its outputs as real paintings, while the discriminator continually refines its ability to detect fakes. Tuning the Adam optimizer was a very time-consuming process, setting learning rates of generator and discriminator at the same level led to a quick mode collapse where the discriminator dominated the generator leading to the generator failing to produce meaningful outputs that were visually distinguishable from purely random noise. The model was allowed to train for 1000 epochs, producing some sample images every 10 epochs to verify its progress.

While there are examples of successful DCGAN Monet generators, after days of tinkering with hyperparameters and architecture, many iterations of DCGAN produced incredibly striking images, but unfortunately none of them resembled Monet paintings to the satisfaction of the authors. As such, the DCGAN method was abandoned, but it is included here because several days of work did go into this pursuit.

Because the Monet dataset is fairly small (300), we applied augmentation to the data. Each image was flipped, mirrored, cropped randomly, and had gaussian noise added, and each of these new images was added to our dataset. This gave our training dataset for the DCGAN a total of 1500 images.

### Results and analysis

As we see in our plot of Generator and Discriminator loss, with the learning rates handicapping the discriminator, the generator was able to start strong, with the discriminator playing catch-up. The plot reveals an interesting dynamic between the two networks. Initially, the generator loss starts around 0.75 and remains relatively stable, while the discriminator loss drops sharply from 2.25 to approximately 1.25 before stabilizing. This suggests that the generator quickly established a strong ability to produce outputs that fooled the discriminator, leading to a relatively balanced adversarial process. With this balance the discriminator did not entirely overpower the generator, possibly allowing some instability in how the generator utilized latent space inputs.

After training, we examined some of the images generated at different epochs. While the generator is producing some very interesting images there are two obvious visual issues. The first is that, while Monet's impressionist style is abstract, they still generally contain visually identifiable objects, people, and landscapes. Our fake Monet images appear purely abstract with no identifiable objects. This is likely because that it's capturing stylistic elements of Monet but is not capable of understanding the representations contained in the images.

The second issue is that from at least epoch 50, the images generated appear identical regardless of the latent input given to our generator. In other words, it seems to just be generating what it "thinks" is a Monet painting without any regard for the input. We attempted to resolve this by systematically adjusting hyperparameters such as learning rates, beta values, and latent dimension sizes. Additionally, we experimented with different architectural changes, including modifying the generator’s initial dense layer and adjusting batch normalization settings. However, despite extensive tuning, the issue persisted. Each attempt led to minimal or no improvement, indicating a deeper structural problem in the training process. Recognizing that incremental changes were unlikely to resolve the fundamental issue, we decided to take a step back and rebuild the model from scratch. This is where our experiment into DCGAN ended and our journey into CycleGAN began.

![image](https://github.com/user-attachments/assets/0510020c-1ac1-46c3-9bcc-96e233a78d03)

![image](https://github.com/user-attachments/assets/7d39ed61-2bfc-4f1e-a7a9-b96f1c779974)

## Model 2: CycleGAN

A Cycle-Consistent Generative Adversarial Network (CycleGAN) architecture is designed to perform image-to-image translation. CycleGAN learns to map between two visual styles, in our case Monet paintings and real photographs. Like all GANs, CycleGANs consist of generators and discriminators in competition with each other. Unlike DCGAN however, CycleGAN has two of each. In our case, one generator learns to translate images from Monet style paintings to real photos and the other from photos to Monet. The "Cycle-Consistent" is the idea that an image translated from one domain and then back to its original domain should be similar to the original image. That is to say a Monet painting converted to a photograph and then back to a Monet image should be visually consistent with the original Monet painting. This allows the CycleGAN model to learn meaningful and reversible tranformations.

The generators follow an encoder-decoder structure with skip connections that help preserve fine details during translation. Each stage of downsampling reduces the image resolution while increasing feature depth, and is followed by group normalization and LeakyReLU activation to help prevent the dying ReLU problem. The upsampling stages reverse this process, using transposed convolutions and dropout to help the model generalize and avoid overfitting.

The discriminators follow a PatchGAN design, which allows them to focus on small sections of an image to determine whether they look real or generated. This helps improve realism at a finer level. Training relies on a combination of losses: an adversarial loss to encourage realistic output, a cycle consistency loss to make sure translations can be reversed cleanly, and an identity loss to discourage unnecessary changes to images that are already in the correct style.

During the course of training, we tried different learning rates and epoch numbers. After testing different hyperparameters the ones that follow were the optimal ones found. All components of the model are optimized using the Adam optimizer with a learning rate of 2e-4 and a β₁ parameter of 0.5, values chosen to stabilize training in GANs. We trained the model for 500 epochs using the datasets of Monet paintings and photographs, allowing the generators to gradually improve their ability to mimic and reverse artistic style.

### Results and Analysis

Looking at the generator and discriminator loss plot, we see an imbalance favoring the discriminator, with its loss staying near zero while the generator loss hovers around 2.5. This suggests the discriminator is too confident in distinguishing real and fake images, which could prevent the generator from receiving useful feedback to improve. The sudden spike at epoch 175 hints at a brief disruption in this balance. While the generator continues learning, the near-zero discriminator loss raises concerns that the model may not be effectively guiding improvements in image diversity. This suggest that Fine-tuning hyperparameters—such as further adjusting the discriminator’s learning rate, introducing dropout, or applying gradient penalty could help improve the model.

As we visually examine some of the photos generated by CycleGAN, we see they are clearly more concrete and less abstract than the beautiful but ethereal images that our DCGAN produced. This is evidently because they are based on real world images rather than random noise, and thus the generator does not need to understand the representations in the image. 

![image](https://github.com/user-attachments/assets/7c9dba2b-bc1c-4d63-89fe-953531439e7d)

![image](https://github.com/user-attachments/assets/5a60eb1f-f2f0-4e8a-b7d1-2a9eac98ef9b)

# References

I used the following tutorials to guide my model design.

DCGAN https://keras.io/examples/generative/dcgan_overriding_train_step/

Cycle GAN https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial/
