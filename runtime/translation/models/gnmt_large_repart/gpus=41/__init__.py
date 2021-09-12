from .gnmt_large_repart import gnmt_large_repart
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2

def arch():
    return "gnmt_large_repart"

def model(criterion):
    return [
        (Stage0(), ["input0", "input1", "input2"], ["out0", "out1"]),
        (Stage1(), ["out0", "out1"], ["out2"]),
        (Stage2(), ["out2"], ["out3"]),
        (criterion, ["out3"], ["loss"])
    ]

def full_model():
    return gnmt_large_repart()
