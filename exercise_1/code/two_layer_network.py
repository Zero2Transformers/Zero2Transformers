from code.boiler_plate.network import Network
from code.single_neuron import SingleNeuron


class SimpleTwoLayerNetwork(Network):
    def __init__(self, w_1, w_2):
        self.w_1 = w_1
        self.w_2 = w_2

        self.layer_1 = SingleNeuron(w_1)
        self.layer_2 = SingleNeuron(w_2)

    def forward(self, X):
        o_1 = self.layer_1.forward(X)
        y_out = self.layer_2.forward(o_1)
        return y_out

    def backward(self, dout):
        gw_2 = self.layer_2.backward(dout)
        gw_1 = self.layer_1.backward(gw_2)
        return gw_1, gw_2
