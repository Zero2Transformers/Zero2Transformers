from code.boiler_plate.network import Network
import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """
    Abstract Base Class for Activation Functions.

    This class serves as a template for building different types of activation functions.
    It defines common attributes and methods that any activation function should have.

    Attributes:
        name (str): Name of the activation function.

    Methods:
        forward: Abstract method for applying the activation in the forward pass.
        backward: Abstract method for computing the gradient of the activation.
    """

    def __init__(self, name: str):
        assert (
            isinstance(name, str) and len(name) > 0
        ), "Name should be a non-empty string."
        self.name = name
        self.cache_grad = False

    def train(self):
        """Enables gradient storage"""
        self.cache_grad = True

    def eval(self):
        """Disables gradient storage"""
        self.cache_grad = False

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the activation function.

        Args:
        - X (np.ndarray): Input data.

        Returns:
        - np.ndarray: Output after applying the activation function.
        """
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the activation function.

        Args:
        - dout (np.ndarray): Upstream gradient.

        Returns:
        - np.ndarray: Gradient of the activation function.
        """
        pass

    def __repr__(self) -> str:
        return f"<Activation(name={self.name})>"


class ReLU(Activation):
    def __init__(self):
        super().__init__("ReLU")

    def forward(self, X):
        batch_size = X.shape[0]

        if self.cache_grad:
            self.cache = X

        # This is confusing, cache the augmanted, and return the acutal X.
        out = np.maximum(X, 0)
        return out

    def backward(self, dout):
        assert self.cache is not None, "run a forward pass before the backward pass"
        X = self.cache
        dX = (X > 0) * dout
        return dX


class Sigmoid(Activation):
    def __init__(self):
        super().__init__("Sigmoid")

    def forward(self, X):
        result = 1 / (1 + np.exp(-X))

        if self.cache_grad:
            self.cache = result

        return result

    def backward(self, dout):
        assert self.cache is not None, "run a forward pass before the backward pass"

        result = self.cache * (1 - self.cache) * dout

        return result
