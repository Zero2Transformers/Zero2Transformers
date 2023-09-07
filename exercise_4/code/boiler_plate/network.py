from abc import ABC, abstractmethod
import numpy as np
import pickle
from code.boiler_plate.weight_initializer import (
    AbstractWeightInitializer,
    AbstractBiasInitializer,
)
from typing import Union


class Network(ABC):
    """
    Abstract Base Class for Neural Network.

    This class serves as a template for building different types of neural networks.
    It defines the common attributes and methods that any network and layer should have.

    Attributes:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        params (dict): Dictionary to store the weights and biases.
        grads (dict): Dictionary to store gradients.
        return_grad (bool): Flag to indicate if gradients should be returned.

    Methods:
        initialize_weights: Abstract method to initialize weights and biases.
        forward: Abstract method for the forward pass.
        backward: Abstract method for the backward pass.
        train: Sets the network in training mode.
        eval: Sets the network in evaluation mode.
        save_weights: Save weights to a file.
        load_weights: Load weights from a file.
    """

    def __init__(
        self,
        input_size,
        output_size,
        weight_strategy: Union[AbstractWeightInitializer, None] = None,
        bias_strategy: Union[AbstractBiasInitializer, None] = None,
    ):
        assert (
            isinstance(input_size, int) and input_size > 0
        ), "input_size should be a positive integer."
        assert (
            isinstance(output_size, int) and output_size > 0
        ), "output_size should be a positive integer."

        self.input_size = input_size
        self.output_size = output_size
        self.params = {}  # Dictionary to store weights and biases
        self.grads = {}  # Dictionary to store gradients
        self.cache_grad = False
        self.initialize_weights(weight_strategy, bias_strategy)

    def initialize_weights(
        self,
        weight_strategy: Union[AbstractWeightInitializer, None] = None,
        bias_strategy: Union[AbstractBiasInitializer, None] = None,
    ):
        """
        Initialize the weight matrix W and biases.

        :param weight_strategy: Optional function that takes the shape of the weight matrix
                                and returns initialized weights.
        :param bias_strategy: Optional function that takes the shape of the bias matrix
                            and returns initialized biases.
        Weights are initialized from a standard normal distribution and scaled by a factor of 0.01
        by default. Biases are initialized to 0.1 by default.
        """

        weight_shape = (self.input_size, self.output_size)
        bias_shape = (self.output_size,)

        if weight_strategy:
            _w = weight_strategy().initialize(weight_shape)
            assert (
                _w.shape == weight_shape
            ), f"Expected weights of shape {weight_shape} but got {self.params['W_weights'].shape}"
        else:
            _w = np.random.randn(*weight_shape) * 0.01

        if bias_strategy:
            b = bias_strategy().initialize(bias_shape)
            assert (
                b.shape == bias_shape
            ), f"Expected biases of shape {bias_shape} but got {b.shape}"
        else:
            b = np.full(bias_shape, 0.1)

        self.params["W"] = np.vstack([b, _w])

    @abstractmethod
    def forward(self, X):
        """
        Forward pass.
        Args:
        - X (np.ndarray): Input data.
        Returns:
        - np.ndarray: Output of the network.
        """
        pass

    @abstractmethod
    def backward(self, dout):
        """
        Backward pass.
        Args:
        - dout (np.ndarray): Gradient of the loss with respect to the output.
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input.
        """
        pass

    def train(self):
        self.cache_grad = True

    def eval(self):
        self.cache_grad = False

    def save_weights(self, filename):
        """
        Save the weights to a file.
        Args:
        - filename (str): The filename to save to.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.params, f)

    def load_weights(self, filename):
        """
        Load weights from a file.
        Args:
        - filename (str): The filename to load from.
        """
        with open(filename, "rb") as f:
            self.params = pickle.load(f)

    def __repr__(self):
        return f"<Network(input_size={self.input_size}, output_size={self.output_size}, params_keys={list(self.params.keys())})>"
