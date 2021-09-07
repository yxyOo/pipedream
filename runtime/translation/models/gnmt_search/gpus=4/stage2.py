import torch
from seq2seq.models.decoder import Classifier
from seq2seq.models.decoder import RecurrentAttention

class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer8 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer9 = torch.nn.Dropout(p=0.2)
        self.layer10 = torch.nn.LSTM(1024, 1024)
        self.layer13 = torch.nn.Dropout(p=0.2)
        self.layer14 = torch.nn.LSTM(1024, 1024)
        self.layer17 = RecurrentAttention(1024, 1024, 1024)
        self.layer20 = torch.nn.Dropout(p=0.2)
        self.layer22 = torch.nn.LSTM(2048, 1024)
        self.layer24 = torch.nn.Dropout(p=0.2)
        self.layer26 = torch.nn.LSTM(2048, 1024)
        self.layer29 = torch.nn.Dropout(p=0.2)
        self.layer31 = torch.nn.LSTM(2048, 1024)
        self.layer34 = Classifier(1024, 32320)

    

    def forward(self, input2, input3, input1):
        out0 = input1.clone()
        out1 = input2.clone()
        out2 = input3.clone()
        out4 = None
        out5 = None
        out6 = None
        out7 = None
        out8 = self.layer8(out2)
        out9 = self.layer9(out0)
        out10 = self.layer10(out9)
        out11 = out10[0]
        out11 = out11 + out0
        out13 = self.layer13(out11)
        out14 = self.layer14(out13)
        out15 = out14[0]
        out15 = out15 + out11
        out17 = self.layer17(out8, out7, out15, out1)
        out18 = out17[2]
        out19 = out17[0]
        out20 = self.layer20(out19)
        out21 = torch.cat([out20, out18], 2)
        out22 = self.layer22(out21, out6)
        out23 = out22[0]
        out24 = self.layer24(out23)
        out25 = torch.cat([out24, out18], 2)
        out26 = self.layer26(out25, out5)
        out27 = out26[0]
        out27 = out27 + out23
        out29 = self.layer29(out27)
        out30 = torch.cat([out29, out18], 2)
        out31 = self.layer31(out30, out4)
        out32 = out31[0]
        out32 = out32 + out27
        out34 = self.layer34(out32)
        return out34
