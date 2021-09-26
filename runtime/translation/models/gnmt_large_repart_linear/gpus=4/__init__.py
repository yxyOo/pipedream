from .gnmt_large_repart_linear import gnmt_large_repart_linear
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

def arch():
    return "gnmt_large_repart_linear"

def model(criterion):
    return [
        (Stage0(), ["input0", "input1", "input2"], ["out2", "out3", "out0"]),
        (Stage1(), ["out2", "out0"], ["out5"]),
        (Stage2(), ["out5", "out2", "out3"], ["out8", "out10"]),
        (Stage3(), ["out8", "out10"], ["out11"]),
        (criterion, ["out11"], ["loss"])
    ]

def full_model():
    return gnmt_large_repart_linear()
