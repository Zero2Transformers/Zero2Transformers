from code.boiler_plate.network import Network
import numpy as np


class FeedForwardNetwork(Network):
    """
    Implementation of a Feed Forward Neural Network which inherits from the Network class.

    This class provides an implementation for Feed Forward Neural Networks, which are
    typically composed of several layers in sequence.

    Attributes:
        layers (list): List of layers added to the network.

    Methods:
        add_layer: Add a layer to the network's list of layers.
    """

    def __init__(self, input_size, output_size):
        self.layers = []
        super().__init__(input_size, output_size)

    def add_layer(self, layer):
        self.layers.append(layer)

    def __repr__(self):
        return " -> ".join([str(layer) for layer in self.layers])

    def initialize_weights(self, _w, _b):
        """Initialize weights and biases. This needs to be implemented by derived classes."""
        for layer in self.layers:
            layer.initialize_weights()

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()

    def forward(self, X):
        """
        Forward pass.
        Args:
        - X (np.ndarray): Input data.
        Returns:
        - np.ndarray: Output of the network.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        """
        Backward pass.
        Args:
        - dout (np.ndarray): Gradient of the loss with respect to the output.
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input.
        """
        # We loop through layers in reverse order during backward pass
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def update_weights(self, learning_rate):
        """
        Update weights for each layer using gradient descent.

        Args:
        - learning_rate (float): The learning rate for the gradient descent update.
        """
        for layer in self.layers:
            if isinstance(layer, Network):
                layer.params["W"] -= learning_rate * layer.grads["W"]
