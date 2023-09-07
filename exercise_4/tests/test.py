from abc import ABC, abstractmethod
import numpy as np
from IPython.display import Image, display
from code.fully_connected import FullyConnected
from code.boiler_plate.weight_initializer import OnesInitializer, ZeroBiasInitializer
from code.simple_dataset import SimpleDataset
from code.mse import MSE


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


class FullyConnectedTest_Forward(UnitTest):
    def __init__(self):
        self.I = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])

        input_size = self.I.shape[1]
        output_size = 3

        self.model = FullyConnected(
            input_size,
            output_size,
            weight_strategy=OnesInitializer,
            bias_strategy=ZeroBiasInitializer,
        )

        self.out_value = self.model.forward(self.I)

    def test(self):
        E_X_1 = np.sum(self.I[0])
        E_X_2 = np.sum(self.I[1])

        if self.out_value.shape != (2, 3):
            return False

        if not np.allclose(self.out_value[0], [E_X_1, E_X_1, E_X_1]):
            return False

        if not np.allclose(self.out_value[1], [E_X_2, E_X_2, E_X_2]):
            return False

        return True


class SimpleDatasetNextTest(UnitTest):
    def __init__(self):
        self.data_x = np.random.rand(103, 5)
        self.data_y = np.random.randint(
            0, 2, size=(103,)
        )  # Generating random labels for demonstration
        self.batch_size = 10

        self.dataset = SimpleDataset(
            data={"x": self.data_x, "y": self.data_y},
            minibatch=True,
            batch_size=self.batch_size,
        )

    def test(self):
        try:
            num_batches = len(self.data_x) // self.batch_size
            for i in range(num_batches):
                batch_x, batch_y = next(self.dataset)
                if batch_x.shape[0] != self.batch_size:
                    return False
                if batch_y.shape[0] != self.batch_size:
                    return False

            last_batch_x, last_batch_y = next(self.dataset)
            expected_last_batch_size = len(self.data_x) % self.batch_size
            if (
                expected_last_batch_size == 0
            ):  # In case the data length is a multiple of the batch size
                expected_last_batch_size = self.batch_size

            if last_batch_x.shape[0] != expected_last_batch_size:
                return False
            if last_batch_y.shape[0] != expected_last_batch_size:
                return False

            # If there's more data, it means there's an error in the __next__ implementation
            next(self.dataset)
            return False

        except StopIteration:  # Expected behavior, so return True
            return True
        except Exception as e:  # Any other error, so return False
            print(e)
            return False


class FullyConnectedBackwardTest(UnitTest):
    def __init__(self):
        self.data = np.random.rand(5, 4)
        self.output_size = 3
        self.model = FullyConnected(input_size=4, output_size=self.output_size)

    def test(self):
        # Perform forward pass
        self.y_pred = self.model.forward(self.data)
        self.y_true = self.y_pred  # Using predicted labels as the true labels

        # Compute loss and gradients
        loss_function = MSE(self.output_size)
        loss = loss_function.forward(self.y_pred, self.y_true)
        dloss = loss_function.backward(self.y_pred, self.y_true)

        # Backward pass through the network
        dx = self.model.backward(dloss)

        # Assert gradients are zero and shapes are correct
        gradients_zero = np.allclose(self.model.grads["W"], 0)
        shapes_correct = self.model.grads["W"].shape == (
            self.data.shape[1] + 1,
            self.output_size,
        )

        return gradients_zero and shapes_correct
