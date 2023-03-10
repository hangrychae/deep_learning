import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h):
        return self.forward(x, h)

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx

        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh

        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx

        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h


        Wrxxt = np.matmul(self.Wrx, x)  # hxd d
        brx = self.brx  # h
        Wrhht_1 = np.matmul(self.Wrh, h) # hxh  h
        brh = self.brh # h
        self.r = self.r_act(Wrxxt + brx + Wrhht_1 + brh)


        Wzxxt = np.matmul(self.Wzx, x)  # hxd  d
        bzx = self.bzx  # h
        Wzhht_1 = np.matmul(self.Wzh, h) # hxh  h
        bzh = self.bzh # h
        self.z = self.z_act(Wzxxt + bzx + Wzhht_1 + bzh)


        Wnxxt = np.matmul(self.Wnx, x)  # hxd  d
        bnx = self.bnx  # h
        rt = self.r 
        self.Wnhht_1 = np.matmul(self.Wnh, h) # hxh  h
        self.bnh = self.bnh # h

        self.n = self.h_act(Wnxxt + bnx + rt*(self.Wnhht_1+self.bnh))


        h_t = (1-self.z)*self.n + self.z*h
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        
        # This code should not take more than 10 lines. 
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        # return h_t
        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
        # This code should not take more than 25 lines.

        x, h = self.x.reshape((1,-1)), self.hidden.reshape((1,-1))
      
        nder, zder, rder = self.h_act.derivative(), self.z_act.derivative(), self.r_act.derivative()
   
        dzt = delta*-self.n + delta*h  
        dnt = delta * (1-self.z)   
        drt = (dnt * nder) * (self.Wnhht_1+self.bnh).T # same with nder.T
       
        self.dWrx = np.matmul((drt * rder.T).T, x)  
        self.dWrx = np.matmul((drt * rder.T).T, x)  
        self.dWzx = np.matmul((dzt * zder.T).T, x) 
        self.dWnx = np.matmul((dnt * nder.T).T, x) 
        self.dWrh = np.matmul((drt * rder.T).T, h)
        self.dWzh = np.matmul((dzt * zder.T).T, h) 
        self.dWnh = np.matmul((dnt * nder * self.r).T, h)       

        self.dbrx = drt * rder.T
        self.dbzx = dzt * zder.T
        self.dbnx = dnt * nder.T
        self.dbrh = drt * rder.T
        self.dbzh = dzt * zder.T
        self.dbnh = dnt * (nder*self.r)    

        dx = np.matmul(dnt * nder, self.Wnx)
        dx += np.matmul(dzt * zder, self.Wzx)
        dx += np.matmul(drt * rder, self.Wrx)

        dh = delta * self.z
        dh += np.matmul(dnt * nder * self.r, self.Wnh)
        dh += np.matmul(dzt * zder, self.Wzh)
        dh += np.matmul(drt * rder, self.Wrh)

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        # return dx, dh
        return dx, dh
