import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2

class gnmt_large_repart(torch.nn.Module):
    def __init__(self):
        super(gnmt_large_repart, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()

    

    def forward(self, input0, input1, input2):
        (out0, out1) = self.stage0(input0, input1, input2)
        out2 = self.stage1(out0, out1)
        out3 = self.stage2(out2)
        return out3
