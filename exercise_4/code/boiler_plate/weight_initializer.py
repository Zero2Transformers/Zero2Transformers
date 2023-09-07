from abc import ABC, abstractmethod
import numpy as np


class AbstractWeightInitializer(ABC):
    @abstractmethod
    def initialize(self, shape):
        pass


class AbstractBiasInitializer(ABC):
    @abstractmethod
    def initialize(self, shape):
        pass


class ZerosInitializer(AbstractWeightInitializer):
    def initialize(self, shape):
        return np.zeros(shape)


class OnesInitializer(AbstractWeightInitializer):
    def initialize(self, shape):
        return np.ones(shape)


class ZeroBiasInitializer(AbstractBiasInitializer):
    def initialize(self, shape):
        biases = np.zeros(shape)
        return biases


class OneBiasInitializer(AbstractBiasInitializer):
    def initialize(self, shape):
        biases = np.ones(shape)
        return biases
