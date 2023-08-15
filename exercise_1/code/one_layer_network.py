from code.boiler_plate.network import Network
from code.single_neuron import SingleNeuron


class SimpleOneLayerNetwork(Network):
    def __init__(self, w_1):
        self.w_1 = w_1

        self.layer_1 = SingleNeuron(w_1)

    def forward(self, X):
        y_out = self.layer_1.forward(X)
        return y_out

    def backward(self, dout):
        gw_1 = self.layer_1.backward(dout)
        return gw_1
