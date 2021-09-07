from .gnmt_search import gnmt_search_partitioned
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2

def arch():
    return "gnmt_search"

def model(criterion):
    return [
        (Stage0(), ["input0", "input1", "input2"], ["out1", "out2", "out0"]),
        (Stage1(), ["out0"], ["out4"]),
        (Stage2(), ["out1", "out2", "out4"], ["out5"]),
        (criterion, ["out5"], ["loss"])
    ]

def full_model():
    return gnmt_search_partitioned()
