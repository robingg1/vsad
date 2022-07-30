import torch 
import torch.nn as nn
import torch.nn.functional as F


class MILModel(nn.Module):
    def __init__(self,input_dim = 1024, hidden_dim1 = 512, hidden_dim2=256,hidden_dim3 = 32, output_dim = 1, drop_p = 0.02):

        super(MILModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 =  nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2,hidden_dim3)
        self.linear4 = nn.Linear(hidden_dim3, output_dim)
        self.dropout = nn.Dropout(drop_p)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight = nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self,x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(self.linear2(x))
        x = self.dropout(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))

        return x
