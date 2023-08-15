from abc import ABC, abstractmethod
import random
from code.activation import ReLU, Sigmoid
import numpy as np
import math
from IPython.display import Image, display


def display_image():
    image_path = "../../the-office-congratulations.jpg"
    display(Image(filename=image_path))


def test_case_runner(test_classes):
    all_tests_passed = True
    for test_class in test_classes:
        all_tests_passed = all_tests_passed and test_class()

    if all_tests_passed:
        display_image()


class UnitTest(ABC):
    def __call__(self):
        try:
            test_passed = self.test()
            if test_passed:
                print(self.define_success_message())
                return True
            else:
                print(self.define_failure_message())
                return False
        except Exception as exception:
            print(self.define_exception_message(exception))
            return False

    @abstractmethod
    def test(self):
        pass

    def define_failure_message(self):
        return "%s failed." % type(self).__name__

    def define_success_message(self):
        return "%s passed." % type(self).__name__

    def define_exception_message(self, exception):
        return "%s failed due to exception: %s." % (type(self).__name__, exception)


class ReLUTest_Forward(UnitTest):
    def __init__(self):
        self.I1 = -1
        self.I2 = 4

        model = ReLU()

        self.value_1 = model.forward(self.I1)
        self.value_2 = model.forward(self.I2)

    def test(self):
        E_V_1 = 0
        E_V_2 = 4

        return self.value_1 == E_V_1 and self.value_2 == E_V_2


class ReLUTest_Backward(UnitTest):
    def __init__(self):
        self.I_1 = -2
        self.I_2 = 2
        self.dout = 3
        self.model = ReLU()

        self.model.forward(self.I_1)
        self.value_1 = self.model.backward(self.dout)

        self.model.forward(self.I_2)
        self.value_2 = self.model.backward(self.dout)

    def test(self):
        E_V_1 = 0
        E_V_2 = self.dout

        return self.value_1 == E_V_1 and self.value_2 == E_V_2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SigmoidTest_Forward(UnitTest):
    def __init__(self):
        self.I1 = -1
        self.I2 = 4

        model = Sigmoid()

        self.value_1 = model.forward(self.I1)
        self.value_2 = model.forward(self.I2)

    def test(self):
        E_V_1 = sigmoid(-1)
        E_V_2 = sigmoid(4)

        return math.isclose(self.value_1, E_V_1) and math.isclose(self.value_2, E_V_2)


class SigmoidTest_Backward(UnitTest):
    def __init__(self):
        self.I_1 = -2
        self.I_2 = 2
        self.dout = 3
        self.model = Sigmoid()

        self.model.forward(self.I_1)
        self.value_1 = self.model.backward(self.dout)

        self.model.forward(self.I_2)
        self.value_2 = self.model.backward(self.dout)

    def test(self):
        E_V_1 = sigmoid(-2) * (1 - sigmoid(-2)) * 3
        E_V_2 = sigmoid(2) * (1 - sigmoid(2)) * 3

        return math.isclose(self.value_1, E_V_1) and math.isclose(self.value_2, E_V_2)
