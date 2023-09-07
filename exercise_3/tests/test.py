from abc import ABC, abstractmethod
import random
from code.mnist_batch_dataset import MNISTDatasetBatch
import numpy as np
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


class MNISTDatasetBatchTest_GetItem(UnitTest):
    def __init__(self):
        root = "data"
        self.dataset_no_memo = MNISTDatasetBatch(
            root=root, batch_size=4, load_to_memory=False
        )
        self.dataset_memo = MNISTDatasetBatch(
            root=root, batch_size=4, load_to_memory=True
        )

        self.index = 1

    def test(self):
        images_1, labels_1 = self.dataset_no_memo[self.index]
        images_2, labels_2 = self.dataset_memo[self.index]
        # expected values are the shapes of the images and labels
        expected_images_shape = (self.dataset_no_memo.batch_size, 28, 28)
        expected_labels_shape = (self.dataset_no_memo.batch_size,)
        return (
            images_1.shape == expected_images_shape
            and images_2.shape == expected_images_shape
            and labels_1.shape == expected_labels_shape
            and labels_1.shape == expected_labels_shape
        )


class MNISTDatasetBatchTest_Len(UnitTest):
    def __init__(self):
        root = "data"
        self.b_1 = 10
        self.d_1 = MNISTDatasetBatch(root=root, batch_size=self.b_1)

        self.b_2 = 7
        self.d_2 = MNISTDatasetBatch(root=root, batch_size=self.b_2)

        self.b_3 = 60000
        self.d_3 = MNISTDatasetBatch(root=root, batch_size=self.b_3)

    def test(self):
        e_1 = int(np.floor(60_000 / self.b_1))
        e_2 = int(np.floor(60_000 / self.b_2))
        e_3 = int(np.floor(60_000 / self.b_3))
        return len(self.d_1) == e_1
