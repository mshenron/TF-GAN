# TF-GAN ![](Media/AI_icon.png)
Easy to use Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf)

to use the code clone the repository to your local system and go to DCGAN directory
and run
```shell
python main.py
```

#### Generative adversarial models (GAN) background:

GAN, introduced by Ian Good fellow in 2014, composed of two deep networks, called Generator and Discriminator that compete with each other. In the course of training, both networks eventually learn how to perform their tasks.
  * Generator: A deep network generates realistic images.
  * Discriminator: A deep network distinguishes real images from computer generated images.

We often compare the GAN networks as a counterfeiter (generator) and a police (discriminator). Initially counterfeiter produces fake Currency and police is trained to identify fake Currency by providing labeled real currency and counterfeit output. However, the same training signal for the police is repurposed for training the counterfeiter giving feedback, why the Currency was fake, counterfitter now tries to print better Currency based on the feedback it received. 

If done correctly, we can lock both parties into competition that eventually the counterfeit is undistinguishable from real money.

#### DCGAN (Deep Convolutional Generative Adversarial Networks):

In DCGAN, we generate an image directly using a generator network while using the discriminator network to guide the generation process.

![](Media/dcgan.png)
[Image source](https://gluon.mxnet.io/chapter14_generative-adversarial-networks/dcgan.html)

The input 'z' to the network is 100-dimensional random vector, which is used to create G(z) i.e. generator output using multiple layers of transpose convolutions (deconvolution)

At the beginning, G(z) are just random noisy images, With the training dataset and the generated images from the generator network, we train the discriminator (an CNN classifier) to classify whether its input image is real or generated. Simultaneously backpropagating the discriminator score to the generator network.
