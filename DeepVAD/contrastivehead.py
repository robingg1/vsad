import torch
import torch.nn as nn

class ContrastiveHead(nn.Module):

    def __init__(self, input_dim = 1024, hidden_dim = 512, hidden_dim1=256,output_dim = 128):
        super(ContrastiveHead,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        # self.linear3 = nn.Linear(hidden_dim1, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)


        return x