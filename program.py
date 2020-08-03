import NeuralNet as Nn
import numpy as np

# now lets actually test our NN code
layer_sizes = (2, 3, 5, 2)
x = np.ones((layer_sizes[0], 1))

net = Nn.NeuralNetwork(layer_sizes)
prediction = net.predict(x)
print(prediction)
