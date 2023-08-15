from code.boiler_plate.loss import Loss


class MSE(Loss):
    def forward(self, y_out, y_truth):
        
        result = None
        
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of the MSE loss function for a single     #
        # training instance.                                                   #
        #                                                                      #
        # y_out represents y^                                                  #
        # y_truth is the label                                                 #
        #                                                                      #
        ########################################################################
        
        pass
        
        ########################################################################
        #                           END OF TODO                                #
        ########################################################################
        
        return result

    def backward(self, y_out, y_truth, n):
        
        result = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass for a single training instance.          #
        # Return the gradient of L w.r.t y_out                                 #
        # n is the number of training instances                                #
        ########################################################################


        pass

        ########################################################################
        #                           END OF TODO                                #
        ########################################################################
        
        return result
