# Tutorial courtesy of Sebastian Lazgue
# https://www.youtube.com/watch?v=8bNIkfRJZpo
import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        # matrices for weights need to be sized based on the sizes of the layers being connected.
        # i.e. if connecting layer with size 4 to layer with size 9, we need to create a 9x4 matrix.
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1::], layer_sizes[:-1])]

        # init weight matrices with random values from bell curve at first.
        # dividing the value we init the weights with to stay contained no matter the input.
        self.weights = [np.random.standard_normal(shape) / shape[1] ** .5 for shape in weight_shapes]

        # init biases with zeroes; column vector for each layer except the input
        self.biases = [np.zeros((shape, 1)) for shape in layer_sizes[1::]]

    def predict(self, input_vector):
        # with a starting the inputs, iterate through all layers in the network,
        # updating the activation at each step, finally giving us our output.
        activation = input_vector
        for weight, bias in zip(self.weights, self.biases):
            activation = self.activation(np.matmul(weight, activation) + bias)

        return activation

    def print_accuracy(self, images, labels):
        predictions = self.predict(images)

        # use list comprehension to generate array of 1's for correct prediction, 0 otherwise.
        number_correct = np.sum([np.argmax(a) == np.argmax(b) for a, b in zip(predictions, labels)])
        print('correct predictions: {0}/{1}; accuracy: {2}%'.format(
            number_correct,
            len(images),
            (number_correct/len(images))*100
        ))

    @staticmethod
    def activation(x):
        # from the tutorial, we're using a sigmoid activation function 1/(1+e^(-x))
        return 1 / (1 + np.exp(-x))
