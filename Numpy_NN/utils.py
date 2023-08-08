import numpy as np
class LeakyRelu:
    def __init__(self):
        pass
    def forward(self, x):
        self.x_last = x
        ret = np.maximum(x, .3 * x)
        return ret
    def backward(self, err, *args, **kwargs):
        return err * ((self.x_last > 0) + (self.x_last < 0) * .3)


class FlattenNp:
    def __init__(self):
        pass
    def forward(self, x):
        # reshape x shape [bs, h, w, filters] to [bs, -1]
        self.bs, self.h, self.w, self.filters = x.shape
        return x.reshape(self.bs, -1)
    
    def backward(self, err, *args, **kwargs):
        # input: [bs, -1]
        # output: [bs, h, w, filters]
        return err.reshape(self.bs, self.h, self.w, self.filters)