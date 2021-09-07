import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer9 = torch.nn.LSTM(2048, 1024)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer9(out0)
        out2 = out1[0]
        return out2
