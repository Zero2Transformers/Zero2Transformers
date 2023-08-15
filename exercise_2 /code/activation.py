from code.boiler_plate.network import Network
import numpy as np


class ReLU(Network):
    def forward(self, X):
        result = None

        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of the ReLU activation function.          #
        # Try using numpy!                                                     #
        #                                                                      #
        # Cache away any value that we could use during backpropagation.       #
        ########################################################################

        pass

        ########################################################################
        #                           END OF TODO                                #
        ########################################################################

        return result

    def backward(self, dout):
        result = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient of L w.r.t X        #
        # Use the value from cache as needed.                                  #
        #                                                                      #
        # dout represents the upstream derivative                              #
        ########################################################################

        pass

        ########################################################################
        #                           END OF TODO                                #
        ########################################################################

        return result


class Sigmoid(Network):
    def forward(self, X):
        result = None

        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of the sigmoid activation function.       #
        #                                                                      #
        # Cache away any value that we could use during backpropagation.       #
        ########################################################################

        pass

        ########################################################################
        #                           END OF TODO                                #
        ########################################################################

        return result

    def backward(self, dout):
        result = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient of L w.r.t X        #
        # Use the value from cache as needed.                                  #
        #                                                                      #
        # dout represents the upstream derivative                              #
        ########################################################################

        pass

        ########################################################################
        #                           END OF TODO                                #
        ########################################################################

        return result
