import numpy as np
from numba import jit
import random
from numba import prange

# @jit(nopython=True)
def conv_f(x_pad, w, bs, h, width, kernel_size, in_shape, out_shape):
    output = np.zeros((bs, h, width, out_shape))
    for i in range(h):
        for j in range(width):
            h_start, w_start = i, j
            h_end, w_end = h_start + kernel_size, w_start + kernel_size
            # batch size, kernel_height, kernel width, in_filters, output_filters
            raw_matmul = x_pad[:, h_start:h_end, w_start:w_end, :, None] * w
            for subh in range(kernel_size):
                for subw in range(kernel_size):
                    for subin in range(in_shape):
                        output[:, i, j, :] += raw_matmul[:,subh, subw, subin]
    return output
@jit(nopython=True)
def optimized_conv_f(x_pad, w, bs, h, width, kernel_size, n_in, out_shape):
    # Adjusting the output shape based on the provided parameters
    output = np.zeros((bs, h, width, out_shape))
    
    # Nested loops for the convolution operation
    for i in range(h):
        for j in range(width):
            for subh in range(kernel_size):
                for subw in range(kernel_size):
                    for subin in range(n_in):
                        for subout in range(out_shape):
                            # Iterating over the batch size as well
                            for b in range(bs):
                                output[b, i, j, subout] += (x_pad[b, i + subh, j + subw, subin] *
                                                            w[subh, subw, subin, subout])
    return output
@jit(nopython=True)
def compute_grad_w(x_last, err, bs, h, w, kernel_size, in_shape, out_shape):
    grad_w = np.zeros((kernel_size, kernel_size, in_shape, out_shape))
    for i in prange(h):
        for j in prange(w):
            for k in prange(kernel_size):
                for l in prange(kernel_size):
                    for m in prange(in_shape):
                        for n in prange(out_shape):
                            for b in prange(bs):
                                grad_w[k, l, m, n] += x_last[b, i + k, j + l, m] * err[b, i, j, n]
    return grad_w

class ConvNp:
    def __init__(self, in_shape, out_shape, l2_reg = 1e-5, initialization = 'kaiming', momentum = 0.0,
                nesterov = False, kernel_size = 3, padding=True):
        if nesterov:
            raise NotImplementedError('nesterov not yet implemented')
        if momentum > 0:
            raise NotImplementedError('momentum not yet implemented')
        self.kernel_size = kernel_size
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.padding = padding
        self.kernel_size = kernel_size
        self.pad_size = int((self.kernel_size - 1)/2)
        self.nesterov = nesterov
        self.l2_reg = l2_reg
        self.momentum = momentum
        self.w = np.random.randn(kernel_size, kernel_size, in_shape, out_shape)
        if initialization == 'xavier':
            self.w = self.w * np.sqrt(6) / np.sqrt(in_shape + out_shape)
        if initialization == 'kaiming':
            self.w = self.w * np.sqrt(2 / in_shape)
            self.b = np.zeros((out_shape))
        else:
            self.b = np.random.randn(out_shape)/out_shape
        if momentum >0:
            self.w_vel = np.zeros_like(self.w)
            self.b_vel = np.zeros_like(self.b)
        
    def forward(self, x):
        # batch_size, h, w, features
        bs, h, w, n_in = x.shape
        x_pad = np.concatenate((np.zeros((bs, 1, w, n_in)), x, np.zeros((bs, 1, w, n_in))), axis=1)
        x_pad = np.concatenate((np.zeros((bs, h+self.pad_size * 2, 1, n_in)), 
                                x_pad, np.zeros((bs, h+self.pad_size * 2, 1, n_in))), axis=2)
        self.x_last = x_pad
#         if self.nesterov:
#             self.w = self.w + self.w_vel * self.momentum
#             self.b = self.b + self.b_vel * self.momentum
        output = conv_f(x_pad, self.w, bs, h, w, self.kernel_size, n_in, self.out_shape)
        # batch, h, w, out_filters
        return output + self.b[None, None, None, :]
    def backward(self, err, lr):
        if 0:
            bs, h, w, n_out = err.shape

            # Rotate the weight kernel by 180 degrees
            rotated_w = np.rot90(self.w, 2, axes=(0, 1))
        
            # Convolve the rotated weight kernel with the error to get the gradient with respect to the input
            pad_size = self.pad_size  # or whatever padding size is appropriate
            err_padded = np.pad(err, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
            
            grad_backward = optimized_conv_f(
                err_padded, rotated_w, bs, h, w, self.kernel_size, n_out, self.in_shape)
        
            # 2. Compute the gradient with respect to the weight kernel
            # Convolve the input with the error to get the gradient with respect to the weight kernel
            grad_w = compute_grad_w(self.x_last, err, bs, h, w, self.kernel_size, self.in_shape, self.out_shape)
            
            # Regularization and weight update
            grad_w = (grad_w - self.l2_reg * self.w) * lr
            beta_update = np.sum(err, axis=(0, 1, 2)) / bs * lr
        
        else:
            bs, h, w, n_out = err.shape
            grad_w = np.zeros((self.kernel_size, self.kernel_size, self.in_shape, self.out_shape))
            e_pad = np.concatenate((np.zeros((bs, 1, w, n_out)), err, np.zeros((bs, 1, w, n_out))), axis=1)
            e_pad = np.concatenate((
                np.zeros((bs, h+self.pad_size * 2, 1, n_out)), e_pad, np.zeros((bs, h+self.pad_size * 2, 1, n_out))
            ), axis=2)
            grad_backward = np.zeros((bs, h, w, self.in_shape))
            for i in range(h):
                for j in range(w):
                    h_start, w_start = i, j
                    h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size
                    # batch size, kernel_height, kernel width, in_filters, output_filters
                    grad_back_raw =  e_pad[:, h_start:h_end, w_start:w_end, None, :] * self.w[None, :, :, :, :]
                    grad_to_input = np.sum(grad_back_raw, axis=(1, 2, 4))
                    # output val shape: batch_size, input_filters
                    grad_backward[:, i, j, :] = grad_to_input
                    
                    grad_w_raw = np.sum(
                        e_pad[:, i:i+1, j:j+1, None, :] * 
                        self.x_last[:, h_start:h_end, w_start:w_end, :, None],
                        axis=0
                    )
                    # kernel height, kernel width, in_filters, out_filters
                    grad_w += grad_w_raw/bs
            
            # err has shape (bs, h, w, out_features)
            # beta_update has shape (out_features)
            beta_update = np.sum(err, axis=(0, 1, 2))
            
        if self.momentum == 0:
            self.w = self.w +  (grad_w - self.l2_reg * self.w) * lr
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