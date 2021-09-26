import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

class gnmt_large_repart_linear(torch.nn.Module):
    def __init__(self):
        super(gnmt_large_repart_linear, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()

    

    def forward(self, input0, input1, input2):
        (out2, out3, out0) = self.stage0(input0, input1, input2)
        out5 = self.stage1(out2, out0)
        (out8, out10) = self.stage2(out5, out2, out3)
        out11 = self.stage3(out8, out10)
        return out11
