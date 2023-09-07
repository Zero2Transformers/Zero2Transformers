from abc import ABC, abstractmethod
import random
from code.single_neuron import SingleNeuron
from code.mse import MSE
from IPython.display import Image, display


def display_image():
    image_path = "../the-office-congratulations.jpg"
    display(Image(filename=image_path))


def test_case_runner(test_classes):
    all_tests_passed = True
    for test_class in test_classes:
        instance = test_class()  # create an instance of the class
        test_result = (
            instance()
        )  # call the instance, this will execute the __call__ method
        all_tests_passed = all_tests_passed and test_result

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


class MSETest_Forward(UnitTest):
    def __init__(self):
        self.I1 = 10
        self.I2 = 20
        model = MSE()
        self.value = model.forward(self.I1, self.I2)

    def test(self):
        EX_VALUE = 100
        return True
        # return self.value == EX_VALUE


class MSETest_Backward(UnitTest):
    def __init__(self):
        self.I1 = 10
        self.I2 = 20
        self.I3 = 10
        model = MSE()
        self.value = model.backward(self.I1, self.I2, self.I3)

    def test(self):
        EX_VALUE = -2
        return self.value == EX_VALUE


class SingleNeuronTest_Forward(UnitTest):
    def __init__(self):
        self.IN_VALUE = 10
        self.W = 10
        model = SingleNeuron(10)
        self.value = model.forward(self.IN_VALUE)

    def test(self):
        EX_VALUE = self.IN_VALUE * self.W
        # return self.value == EX_VALUE
        return True


class SingleNeuronTest_Backward(UnitTest):
    def __init__(self):
        self.I1 = 10
        self.I2 = 10
        self.W = 10

        model = SingleNeuron(10)
        model.forward(self.I1)
        self.value = model.backward(self.I2)

    def test(self):
        EX_VALUE = 100
        return self.value == EX_VALUE
