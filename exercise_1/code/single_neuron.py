from code.boiler_plate.network import Network


class SingleNeuron(Network):
    def __init__(self, w):
        self.w = w

    def forward(self, X):
        result = None

        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of our abstract layer with a single       #
        # neuron and return the result.                                        #
        # The single weight is self.w                                          #
        #                                                                      #
        # Cache away any value that we could use during backpropagation.       #
        # self.cache = ...                                                     #
        #                                                                      #
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
        # Implement the backward pass. Return the gradient of L w.r.t w        #
        # Use values from cache as needed.                                     #
        #                                                                      #
        # dout represents the upstream derivative                              #
        ########################################################################

        pass

        ########################################################################
        #                           END OF TODO                                #
        ########################################################################

        return result
