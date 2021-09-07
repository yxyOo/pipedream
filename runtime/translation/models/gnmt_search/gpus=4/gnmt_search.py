import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2

class gnmt_search_partitioned(torch.nn.Module):
    def __init__(self):
        super(gnmt_search_partitioned, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()

    

    def forward(self, input0, input1, input2):
        (out1, out2, out0) = self.stage0(input0, input1, input2)
        out4 = self.stage1(out0)
        out5 = self.stage2(out1, out2, out4)
        return out5
