import torch
from seq2seq.models.decoder import Classifier

class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.layer4 = torch.nn.Dropout(p=0.2)
        self.layer6 = torch.nn.LSTM(2048, 1024)
        self.layer9 = torch.nn.Dropout(p=0.2)
        self.layer11 = torch.nn.LSTM(2048, 1024)
        self.layer14 = Classifier(1024, 32320)

    

    def forward(self, input4, input3):
    # def forward(self, input4, input3, input2):
        out0 = input3.clone()
        out1 = input4.clone()
        # out2 = input2.clone()
        out2 = None
        out3 = None
        out4 = self.layer4(out0)
        out5 = torch.cat([out4, out1], 2)
        out6 = self.layer6(out5, out2)
        out7 = out6[0]
        out7 = out7 + out0
        out9 = self.layer9(out7)
        out10 = torch.cat([out9, out1], 2)
        out11 = self.layer11(out10, out3)
        out12 = out11[0]
        out12 = out12 + out7
        out14 = self.layer14(out12)
        return out14
