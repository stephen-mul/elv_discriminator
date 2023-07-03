import torch

class ohe():
    def __init__(self, n_vals=10):
        self.vec = torch.zeros(n_vals, dtype=torch.float16)

    def encode(self, integer_value):
        self.vec[integer_value]=1.0
        return self.vec
