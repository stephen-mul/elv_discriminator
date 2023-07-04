import torch

class ohe():
    def __init__(self, n_vals=10):
        self.n_vals = n_vals

    def encode(self, input):
        length = list(input.shape)[0]
        vecs = torch.zeros(length, self.n_vals, dtype=torch.float16)
        count = 0
        for integer in input:
            vecs[count, integer] = 1.0
            count += 1
        return vecs
