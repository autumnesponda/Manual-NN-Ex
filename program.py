import NeuralNet as Nn
import numpy as np
import matplotlib.pyplot as plot

# import data of handwritten numbers
# training_images are the numbers themselves in greyscale (values from 0 to 1)
# training_labels are ground truths for each image in training_images
# SOURCE: https://github.com/SebLague/Mnist-data-numpy-format
with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

# how to show an image from dataset with pyplot
# plot.imshow(training_images[0].reshape(28, 28), cmap='gray')
# plot.show()

# now lets actually configure the neural net !
# 1 input layer of size 784; the images are in a 28x28 grid = 764 pixels
# 1 hidden layer with 15 neurons
# 1 output layer of size 10; which number the net thinks it sees (0 to 9)
layer_sizes = (784, 15, 10)

# generate a net with the dimensions we specified above
# + test accuracy of untrained network
net = Nn.NeuralNetwork(layer_sizes)
net.print_accuracy(training_images, training_labels)

# accuracy of untrained net is ~10% ± 2%.
# aka, the net is no better than random guessing.

# TODO: train the neural net
