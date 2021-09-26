import torch
from seq2seq.models.decoder import RecurrentAttention

class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer7 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer8 = torch.nn.Dropout(p=0.2)
        self.layer9 = torch.nn.LSTM(1024, 1024)
        self.layer12 = RecurrentAttention(1024, 1024, 1024)
        self.layer15 = torch.nn.Dropout(p=0.2)
        self.layer17 = torch.nn.LSTM(2048, 1024)

    

    def forward(self, input1, input2, input3):
        out0 = input1.clone()
        out1 = input2.clone()
        out2 = input3.clone()
        # out4 = None
        out5 = None
        out6 = None
        out7 = self.layer7(out2)
        out8 = self.layer8(out0)
        out9 = self.layer9(out8)
        out10 = out9[0]
        out10 = out10 + out0
        out12 = self.layer12(out7, out6, out10, out1)
        out13 = out12[2]
        out14 = out12[0]
        out15 = self.layer15(out14)
        out16 = torch.cat([out15, out13], 2)
        out17 = self.layer17(out16, out5)
        out18 = out17[0]
        return (out13, out18)
        # return (out13, out18, out4)
