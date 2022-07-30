import torch
import torch.nn as nn

class FinalHead(nn.Module):

    def __init__(self, input_dim = 128, output_dim = 1, drop_p = 0.0):
        super(FinalHead,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        
        x = torch.sigmoid(self.dropout(self.linear1(x)))

        return x