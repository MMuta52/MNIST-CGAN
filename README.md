# MNIST CGAN

My first GAN! Initially made as a basic GAN, but adapted it to be a **conditional GAN**. This way, the GAN can take a digit as input and generate whatever number I want!

Trained on [MNIST Handwritten Digits Dataset](http://yann.lecun.com/exdb/mnist/).

I also used [Google Colab](https://colab.research.google.com) when experimenting with training. Great way to access a GPU for small projects like this!

Used [this PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) and [BeierZhu's GAN implementation](https://github.com/BeierZhu/GAN-MNIST-Pytorch) for guidance.

# Files

 **cgan** is where the CGAN is trained. It automatically loads in the MNIST dataset - no download required!

**Discriminator** holds the Disciminator class

**Generator** holds the Generator class

**app** holds a little bit of logic to take any integer as input and generate a handwritten-looking image.

# Dependencies

[PyTorch](https://pytorch.org) - Neural network training
[Numpy](https://numpy.org) - Math
[PIL](https://pillow.readthedocs.io/en/stable/) - Displaying images
[Matplotlib](https://matplotlib.org) - Displaying images
