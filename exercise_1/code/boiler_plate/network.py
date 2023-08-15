from abc import ABC, abstractmethod

class Network(ABC):
    @abstractmethod
    def forward(self, X):
        return NotImplementedError

    @abstractmethod
    def backward(self, dout):
        return NotImplementedError
