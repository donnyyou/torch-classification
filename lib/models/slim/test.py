import torch

class My(torch.nn.Module):
    
    def __init__(self):
        super(My, self).__init__()
        self.add_module('', torch.nn.Conv2d(3, 1, 1))


my = My()
print(my.state_dict())

