# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y

        # Defining constant for log exp trick. 
        # According to blog, max of logits works best as c
        c = np.amax(self.logits,axis=1)
        
        # Calculating log exponent sum
        temp = np.log(np.sum(np.exp(self.logits-c[:,None]),axis=1)) + c
        
        # Calculating log of softmax
        log_softmax = self.logits - temp[:,None]
        
        # Calculating softmax to calculate derivative in future
        self.softmax = np.exp(log_softmax)
        
        # Calculating cross entropy 
        self.CE = -1*np.sum(np.multiply(self.labels,log_softmax), axis=1)
        
        return self.CE

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        #raise NotImplemented
        # reffered from spring 20CMU DL
        return self.softmax - self.labels
