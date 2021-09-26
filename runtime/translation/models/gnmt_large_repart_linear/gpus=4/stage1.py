import torch
from seq2seq.models.encoder import EmuBidirLSTM

class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer2 = EmuBidirLSTM(1024, 1024)
        self.layer3 = torch.nn.Dropout(p=0.2)
        self.layer4 = torch.nn.LSTM(2048, 1024)
        self.layer6 = torch.nn.Dropout(p=0.2)
        self.layer7 = torch.nn.LSTM(1024, 1024)

    

    def forward(self, input1, input0):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = self.layer2(out0, out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = out4[0]
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = out7[0]
        out8 = out8 + out5
        return out8
