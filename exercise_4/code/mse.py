from code.boiler_plate.network import Network
import numpy as np


class MSE(Network):
    def __init__(self, input_size, individual_losses=False):
        self.individual_losses = individual_losses
        super().__init__(input_size, output_size=input_size if individual_losses else 1)

    def forward(self, y_out, y_truth):
        """
        Performs the forward pass of the Element-wise MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return:
            - individual_losses=False --> A single scalar, which is the mean of the MSE loss
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of MSE loss for each sample of your training set.
        """
        result = (y_out - y_truth) ** 2

        if self.individual_losses:
            return result
        return np.mean(result)

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the Element-wise MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss gradients w.r.t y_out for each sample of your training set.
        """
        gradient = 2 * (y_out - y_truth)
        return gradient
