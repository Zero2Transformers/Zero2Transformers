from code.boiler_plate.network import Network
import numpy as np


class FullyConnected(Network):
    """
    A simple Fully Connected layer, characterized by a weight matrix W.
    The layer performs the following operation: o = X * W
    """

    def forward(self, X):
        """
        Performs the forward pass of the layer.

        :param X: Input data of shape (batch_size, input_size).
                  Each row corresponds to a single data point.
        :return: Output of the layer after applying the linear transformation, shape (batch_size, output_size).
        """
        assert self.params["W"] is not None, "weight matrix W is not initialized"

        batch_size = X.shape[0]

        out = None
        self.cache = None
        X_augmented = None

        ########################################################################
        # TODO:                                                                #
        # 1. Augment the input data (X) with a column of ones for the bias.    #
        # 2. Perform the linear transformation using the augmented input and   #
        #    weight matrix W. Store the result in 'out'.                       #
        # 3. Cache the augmented input for use in the backward pass, but only  #
        #    when needed!                                                      #
        ########################################################################


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return out

    def backward(self, dout):
        """
        Performs the backward pass of the layer.

        :param dout: Gradient of the loss with respect to the output of this layer, shape (batch_size, output_size).
        :return: Gradient of the loss with respect to the input of this layer, shape (batch_size, input_size).
        """
        assert self.cache is not None, "run a forward pass before the backward pass"

        # Retrieving cached input
        (X,) = self.cache

        self.grads["W"] = None
        dX = None

        ########################################################################
        # TODO:                                                                #
        # 1. Compute the gradient with respect to the weight matrix W.         #
        # 2. Compute the gradient with respect to the input X, excluding the   #
        #    gradient contributions from the augmented bias column.            #
        ########################################################################


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.cache = None

        return dX
