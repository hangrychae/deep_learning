import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        stride = 1

        output_size_w = (self.A.shape[2]-self.kernel)//stride + 1
        output_size_h= (self.A.shape[3]-self.kernel)//stride + 1
        Z = np.zeros((self.A.shape[0], self.A.shape[1], output_size_w, output_size_h))
        self.max_index = np.zeros((self.A.shape[0], self.A.shape[1], output_size_w, output_size_h), dtype=object)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                for k in range(Z.shape[2]):
                    for l in range(Z.shape[3]):
                        A_sub = A[i,j,k:k+self.kernel, l:l+self.kernel]
                        ind = np.unravel_index(A_sub.argmax(), A_sub.shape)
                        Z[i,j,k,l] = np.max(A_sub)
                        tmp = A_sub.shape[0]*ind[0]+ind[1]
                        self.max_index[i,j,k,l] = (k+tmp//self.kernel)*A.shape[2] + l+tmp%self.kernel
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)
        for i in range(self.max_index.shape[0]):
            for j in range(self.max_index.shape[1]):
                for k in range(self.max_index.shape[2]):
                    for l in range(self.max_index.shape[3]):
                        max_index = self.max_index[i,j,k,l]
                        r = max_index // dLdA.shape[2]
                        c = max_index % dLdA.shape[3]
                        dLdA[i,j,r,c] += dLdZ[i,j,k,l]
                        
        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        stride = 1

        output_size_w = (self.A.shape[2]-self.kernel)//stride + 1
        output_size_h= (self.A.shape[3]-self.kernel)//stride + 1
        Z = np.zeros((self.A.shape[0], self.A.shape[1], output_size_w, output_size_h))
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                for k in range(Z.shape[2]):
                    for l in range(Z.shape[3]):
                        A_sub = A[i,j,k:k+self.kernel, l:l+self.kernel]
                        Z[i,j,k,l] = np.mean(A_sub)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)

        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for k in range(dLdZ.shape[2]):
                    for l in range(dLdZ.shape[3]):
                        tmp = dLdZ[i,j,k,l]
                        dLdA[i,j,k:k+self.kernel,l:l+self.kernel] += np.sum(tmp)/(self.kernel**2)
                        
        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        pooled = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(pooled)
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        down_dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(down_dLdZ)

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        pooled = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(pooled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        down_dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(down_dLdZ)
        return dLdA
