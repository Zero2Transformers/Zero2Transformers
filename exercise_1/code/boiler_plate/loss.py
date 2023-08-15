from abc import ABC, abstractmethod

class Loss(object):
    @abstractmethod
    def forward(self, y_out, y_truth):
        return NotImplementedError
    @abstractmethod
    def backward(self, y_out, y_truth, upstream=1.0):
        return NotImplementedError
