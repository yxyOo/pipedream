# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.decoder import RecurrentAttention

class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        # self.layer6 = torch.nn.LSTM(1024, 1024)
        # self.layer9 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        # self.layer11 = torch.nn.Dropout(p=0.2)
        # self.layer12 = torch.nn.LSTM(1024, 1024)
        self.layer15 = RecurrentAttention(1024, 1024, 1024)

    def forward(self, input0, input3, input1):
       
        # out15 = self.layer15(out9, out5, out13, out2)
        out10 = [None, None, None, None]  # out4 is hidden, might need to be initialized differently.
        # out11 = out10[0]
        out9=input0.clone()
        out5=out10[0]
        out13=input3.clone()
        out2=input1.clone()
        out15 = self.layer15(out9, out5, out13, out2)
        out16 = out15[0]
        return (out15[2], out16)

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
