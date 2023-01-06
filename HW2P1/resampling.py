from re import A
import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        new_size = A.shape[2] * self.upsampling_factor - (self.upsampling_factor - 1)
        Z = np.zeros((A.shape[0], A.shape[1], new_size))
        # should it be np.ones((size), dtype=int)? 
        Z[:,:,::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        dLdA = dLdZ[:,:,::self.upsampling_factor]

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        Z = A[:,:,::self.downsampling_factor]
        self.A_size = A.shape
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = np.zeros(self.A_size)
        dLdA[:,:,::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        new_size_w = A.shape[2] * self.upsampling_factor - (self.upsampling_factor - 1)
        new_size_h = A.shape[3] * self.upsampling_factor - (self.upsampling_factor - 1)
  
        Z = np.zeros((A.shape[0], A.shape[1], new_size_w, new_size_h))
        # should it be np.ones((size), dtype=int)? 
        Z[:,:,::self.upsampling_factor,::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = dLdZ[:,:,::self.upsampling_factor,::self.upsampling_factor]

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        self.A_size = A.shape
        Z = A[:,:,::self.downsampling_factor,::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
    
        #print(self.A_size)
        #print(self.)
        dLdA = np.zeros(self.A_size)
        # should it be np.ones((size), dtype=int)? 
        dLdA[:,:,::self.downsampling_factor,::self.downsampling_factor] = dLdZ

        return dLdA