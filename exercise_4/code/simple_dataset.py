from code.boiler_plate.dataset import Dataset
import numpy as np


class SimpleDataset(Dataset):
    """
    SimpleDataset derived from the Dataset class.

    This dataset handles data provided as numpy arrays and corresponding labels.
    It can also return minibatches of the data if needed.

    Attributes:
        data (dict): Dictionary containing 'x' for data and 'y' for labels, both as numpy arrays.
        minibatch (bool): If True, returns minibatches of data and labels.
        batch_size (int): Size of each minibatch.
        current_index (int): Index to keep track of data when fetching minibatches.
    """

    def __init__(self, data, minibatch=False, batch_size=None):
        if not all(key in data for key in ["x", "y"]):
            raise ValueError("Data dictionary must have 'x' and 'y' keys.")

        if not isinstance(data["x"], np.ndarray) or not isinstance(
            data["y"], np.ndarray
        ):
            raise ValueError("Both data and labels should be numpy arrays.")

        self.data = data

        self.minibatch = minibatch
        if minibatch:
            if batch_size is None or batch_size <= 0:
                raise ValueError(
                    "batch_size must be a positive integer if minibatch is set to True."
                )
            self.batch_size = batch_size

        self.current_index = 0  # For tracking current index in minibatch fetching

    def __getitem__(self, index):
        """Returns the data point and its corresponding label at the given index."""
        return self.data["x"][index], self.data["y"][index]

    def __len__(self):
        """Returns the number of data points."""
        return len(self.data["x"])

    def __iter__(self):
        """Returns the iterator object (self)."""
        self.shuffle()
        self.current_index = 0  # Resetting for fresh iteration
        return self

    def __next__(self):
        """Returns the next item (data, label) or minibatch based on configuration."""

        ########################################################################
        # TODO:                                                                #
        # 1. Handle the case when minibatch is not used:                       #
        #    - The output should be a tuple (data_point, label).               #
        #    - Each `data_point` should be an individual sample from data["x"].#
        #    - `label` should be the corresponding label from data["y"].       #
        #                                                                      #
        # 2. Check for the end of the data list:                               #
        #    - If you reach the end of the data list, raise StopIteration.     #
        #                                                                      #
        # 3. Fetch the data and labels using appropriate slicing:              #
        #    - If minibatch is used, return a tuple containing two arrays.     #
        #    - The first array should contain `batch_size` samples from        #
        #      data["x"] starting from `current_index`.                        #
        #    - The second array should be the corresponding labels from        #
        #      data["y"].                                                      #
        #    - If the remaining data is less than `batch_size`, return all the #
        #      remaining data and labels.                                      #
        ########################################################################

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def shuffle(self):
        """
        Shuffles the rows of the dataset and its labels in a consistent manner.
        """
        indices = np.arange(self.data["x"].shape[0])
        np.random.shuffle(indices)
        self.data["x"] = self.data["x"][indices]
        self.data["y"] = self.data["y"][indices]

    def __str__(self):
        """
        Returns a string representation of the dataset and labels.
        """
        return f"Data:\n{self.data['x']}\nLabels:\n{self.data['y']}"
