import numpy as np
class DenseNp:
    def __init__(self, in_shape, out_shape, l2_reg = 1e-5, initialization = 'xavier', momentum = 0.0,
                nesterov = False):
        self.nesterov = nesterov
        self.l2_reg = l2_reg
        self.momentum = momentum
        self.w = np.random.randn(in_shape, out_shape)
        if initialization == 'xavier':
            self.w = self.w * np.sqrt(6) / np.sqrt(in_shape + out_shape)
        self.b = np.random.randn(out_shape)/out_shape
        if momentum >0:
            self.w_vel = np.zeros_like(self.w)
            self.b_vel = np.zeros_like(self.b)
        
    def forward(self, x):
        self.x_last = x
        if self.nesterov:
            self.w = self.w + self.w_vel * self.momentum
            self.b = self.b + self.b_vel * self.momentum
        ret = x @ self.w + self.b
        return ret
    def backward(self, err, lr):
        if len(err.shape) < 2:
            err = err[:, None]
        grad_backward = err @ self.w.T
        if self.momentum == 0:
            self.w = self.w +  ((self.x_last.T @ err) - self.l2_reg * self.w) * lr
            # err has shape (bs, out_features)
            # beta_update has shape (out_features)
            beta_update = err.sum(0)
            self.b = self.b + beta_update * lr
        else:
            self.w_vel = self.w_vel * self.momentum +  (
                (self.x_last.T @ err) - self.l2_reg * self.w) * (1- self.momentum) * lr
            self.b_vel = self.b_vel * self.momentum + err.sum(0) * (1- self.momentum) * lr
            if self.nesterov:
                self.w += ((self.x_last.T @ err) - self.l2_reg * self.w) * (1- self.momentum) * lr
                self.b += err.sum(0) * (1- self.momentum) * lr
            else:
                self.w += self.w_vel
                self.b += self.b_vel

        return grad_backward