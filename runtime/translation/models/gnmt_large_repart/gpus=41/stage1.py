import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer28 = torch.nn.Dropout(p=0.2)
        self.layer30 = torch.nn.LSTM(2048, 1024)
        self.layer32 = torch.nn.Dropout(p=0.2)
        self.layer34 = torch.nn.LSTM(2048, 1024)
        self.layer37 = torch.nn.Dropout(p=0.2)
        self.layer39 = torch.nn.LSTM(2048, 1024)

    

    def forward(self,  input3, input2):
        out6 = None
        out5 = None
        out4 = None
        out26=input3.clone()
        out27=input2.clone()
        out28 = self.layer28(out27)
        out29 = torch.cat([out28, out26], 2)
        out30 = self.layer30(out29, out6)
        out31 = out30[0]
        out32 = self.layer32(out31)
        out33 = torch.cat([out32, out26], 2)
        out34 = self.layer34(out33, out5)
        out35 = out34[0]
        out35 = out35 + out31
        out37 = self.layer37(out35)
        out38 = torch.cat([out37, out26], 2)
        out39 = self.layer39(out38, out4)
        out40 = out39[0]

        # out0 = input0.clone()
        # out1 = input1.clone()
        out0 = out35 + out40
        return out0
