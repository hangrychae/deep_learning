# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        stride = 1
        output_size = (self.A.shape[2]-self.kernel_size)//stride + 1
        Z = np.zeros((self.A.shape[0], self.out_channels, output_size))

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                for k in range(Z.shape[2]):
                    Am = A[i,:,k:k+self.kernel_size]
                    Wm = self.W[j,:,:]
                    Z[i,j,k] = np.sum(Am*Wm)
                Z[i,j] += self.b[j]

        #Z = Z + self.b
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        self.dLdb = np.sum(np.sum(dLdZ, axis=2), axis=0)

        # same number of rows as weight
        broadcast_dLdZ = np.tile(dLdZ, (self.A.shape[1]//dLdZ.shape[1],1)) 
        #print(self.dLdW.shape)  # (5,10,4)
        #print(dLdZ.shape) # (2,5,200)

        # dw shape = outchannel inchannel kernel size 
        for i in range(dLdZ.shape[0]): # batch_size 
            for j in range(dLdZ.shape[1]): # out_channels
                for k in range(dLdZ.shape[2]): # output_size
                    for l in range(self.in_channels):
                        for m in range(self.kernel_size):
                            Am = self.A[i, l, k+m] 
                            dLdZm = dLdZ[i, j, k]
                            self.dLdW[j, l, m] += Am * dLdZm

       

        # padded left and right 
        padded_dLdZ = np.pad(dLdZ, ((0,0),(0,0),(self.kernel_size-1, self.kernel_size-1)), 'constant', constant_values=0)
        flipped_W = self.W[:,:,::-1]

        dLdA = np.zeros(self.A.shape)

        # dldz (batch_size, out_channels, output_size)
        # w (out_channels, in_channels, kernel_size)
        
        
        output_size = (self.A.shape[2]-self.kernel_size) + 1
        input_size = output_size - 1 + self.kernel_size

        for i in range(dLdZ.shape[0]): # batch_size          
            for j in range(self.W.shape[0]): # out_channels
                for k in range(self.W.shape[1]): # in_channels
                    for l in range(self.W.shape[2]-1, -1, -1): # kernel_size
                        for m in range(output_size): # output_size 
                        #for m in range(self.A.shape[2]): # input_size 
                            dLdZm = dLdZ[i, j, m] 
                            Wm = self.W[j, k, l]
                            input_pos = l + m
                            dLdA[i,k,input_pos] += dLdZm * Wm

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size)
        self.downsample1d = Downsample1d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        # TODO
        result = self.conv1d_stride1.forward(A)
        result = self.downsample1d.forward(result)
        # downsample
        Z = result

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        result = self.downsample1d.backward(dLdZ)
        result = self.conv1d_stride1.backward(result)
        
        # Call Conv1d_stride1 backward
        dLdA = result

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        stride = 1
        output_size_w = (self.A.shape[2]-self.kernel_size)//stride + 1
        output_size_h= (self.A.shape[3]-self.kernel_size)//stride + 1

        Z = np.zeros((self.A.shape[0], self.out_channels, output_size_w, output_size_h))

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                for k in range(Z.shape[2]):
                    for l in range(Z.shape[3]):
                        Am = A[i,:,k:k+self.kernel_size, l:l+self.kernel_size]
                        Wm = self.W[j,:,:,:]
                        Z[i,j,k,l] = np.sum(Am*Wm)
                Z[i,j] += self.b[j]


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # dw shape = outchannel inchannel kernel size kernel size

        for i in range(dLdZ.shape[0]): # batch_size 
            for j in range(dLdZ.shape[1]): # out_channels
                for k in range(dLdZ.shape[2]): # output_width
                    for l in range(dLdZ.shape[3]): # output_height
                        for m in range(self.in_channels):
                            for n1 in range(self.kernel_size):
                                for n2 in range(self.kernel_size):
                                    Am = self.A[i, m, k+n1, l+n2] 
                                    dLdZm = dLdZ[i, j, k, l]
                                    self.dLdW[j, m, n1, n2] += Am * dLdZm

        self.dLdb = np.sum(dLdZ, (0,2,3))


        dLdA = np.zeros(self.A.shape)
        output_w = (self.A.shape[2]-self.kernel_size) + 1
        output_h = (self.A.shape[3]-self.kernel_size) + 1

 
        for i in range(dLdZ.shape[0]): # batch_size          
            for j in range(self.W.shape[0]): # out_channels
                for k in range(self.W.shape[1]): # in_channels
                    for l1 in range(self.W.shape[2]-1, -1, -1): # kernel_size
                        for l2 in range(self.W.shape[3]-1, -1, -1): # kernel_size
                            for m in range(output_w): # output_width
                                for n in range(output_h): # output_height
                                    dLdZm = dLdZ[i, j, m, n] 
                                    Wm = self.W[j, k, l1, l2]
                                    dLdA[i,k,l1+m,l2+n] += dLdZm * Wm
        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size)
        self.downsample2d = Downsample2d(self.stride)


    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        result = self.conv2d_stride1.forward(A)
        result = self.downsample2d.forward(result)
        # downsample
        Z = result

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        result = self.downsample2d.backward(dLdZ)
        result = self.conv2d_stride1.backward(result)
        
        # Call Conv1d_stride1 backward
        dLdA = result

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(self.upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A)
        print(A.shape)
        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)
        print(Z.shape)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ)
        print(dLdZ.shape)
        dLdA =  self.upsample1d.backward(delta_out)
        print(dLdA.shape)
        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size)
        self.upsample2d = Upsample2d(self.upsampling_factor)

    
    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A)

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ)

        dLdA = self.upsample2d.backward(delta_out)

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """

        self.A_shape = A.shape
        Z = np.reshape(A, (A.shape[0], A.shape[1]*A.shape[2]))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = np.reshape(dLdZ, self.A_shape)

        return dLdA

