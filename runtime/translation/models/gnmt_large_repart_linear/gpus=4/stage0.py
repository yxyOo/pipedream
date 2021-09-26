import torch


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer6 = torch.nn.Embedding(32320, 1024, padding_idx=0)

    

    def forward(self, input0, input1, input2):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input2.clone()
        out6 = self.layer6(out0)
        return (out1, out2, out6)
