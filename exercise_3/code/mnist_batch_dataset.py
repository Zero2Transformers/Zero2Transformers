from code.mnist_dataset import MNISTDataset
import numpy as np


class MNISTDatasetBatch(MNISTDataset):
    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def __getitem__(self, index):
        images, labels = None, None
        ########################################################################
        # TODO:                                                                #
        # Implement the __getitem__ method for batch loading.                  #
        # - 'index' is the index of the batch.                                 #
        # - You should return a batch of images and a batch of labels.         #
        # - If 'self.load_to_memory' is True, the images and labels are        #
        #   already loaded in 'self.images' and 'self.labels'.                 #
        # - Otherwise, you should read the images and labels from the file     #
        #   system. You can use the '__getitem__' method of the super class    #
        #   to read a single image and label.                                  #
        # - The return type be a two numpy array of size                       #
        #  (self.batch_size,28,28) and (self.batch_size, )                     #
        #                                                                      #
        # Hint:                                                                #
        # - Calculate 'start' and 'end' indices of the images and labels in    #
        #   the current batch                                                  #
        ########################################################################

        pass

        ########################################################################
        #                           END OF TODO                                #
        ########################################################################

        return np.array(images), np.array(labels)

    def __len__(self):
        result = None
        ########################################################################
        # TODO:                                                                #
        # Implement the __len__ method for batch loading.                      #
        # - You should return the total number of batches in the dataset.      #
        # - Keep in mind that the number of images might not be divisible by   #
        #  the batch size. We want to ignore the extra images.                 #
        # Hint:                                                                #
        # - Use the 'super' function to call the '__len__' method of the       #
        #   super class to get the total number of images and labels in the    #
        #   dataset.                                                           #
        ########################################################################

        pass

        ########################################################################
        #                           END OF TODO                                #
        ########################################################################

        return result
