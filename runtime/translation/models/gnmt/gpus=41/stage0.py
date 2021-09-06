# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.encoder import EmuBidirLSTM

class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer4 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer5 = EmuBidirLSTM(1024, 1024)
        self.layer6 = torch.nn.Dropout(p=0.2)
        self.layer7 = torch.nn.LSTM(2048, 1024)
        self.layer9 = torch.nn.Dropout(p=0.2)
        
        self.layer10 = torch.nn.LSTM(1024, 1024)
        self.layer11 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer12 = torch.nn.Dropout(p=0.2)
        self.layer13 = torch.nn.LSTM(1024, 1024)

    def forward(self, input0, input1,input2):
        out0 = input0.clone()
        out1 = input1.clone()
        out18 = input2.clone()
        out4 = self.layer4(out0)
        out5 = self.layer5(out4, out1)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = out7[0]
        out9 = self.layer9(out8)
        # (Stage0(), ["input0", "input1"], ["out2", "out1"]),
        # (Stage1(), ["out2", "input1", "input2", "out1"], ["out3", "out7"]),
        # input0, input2, input3, input1
        # out0=out9,out1=out8,out2=input1,out3=input2
        # out0 = input0.clone()
        # out1 = input1.clone()
        # out2 = input2.clone()
        # out3 = input3.clone()
        # out10 = [None, None, None, None]  # out4 is hidden, might need to be initialized differently.
        # out11 = out10[0]
        out12 = self.layer10(out9)
        out13 = out12[0]
        out14 = self.layer11(out18)
        out13 = out13 + out8
        out15 = self.layer12(out13)
        out16 = self.layer13(out15)
        out17 = out16[0]
        out17 = out17 + out13
        #14 11 17 input1
        
        # return (out9, out8)
        # return (out14,out11,out17,out18)
        return (out14,out17,out1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
