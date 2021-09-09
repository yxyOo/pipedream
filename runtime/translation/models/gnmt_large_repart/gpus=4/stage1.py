import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        

    

    def forward(self, input1, input0):
        out0 = input0.clone()
        out1 = input1.clone()
        out0 = out0 + out1
        return out0
