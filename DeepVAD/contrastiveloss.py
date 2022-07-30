import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class ContrastiveLoss(nn.Module):

    def __init__(self, temperature = 0.5, batch_size = config.TRAIN_BATCH_SIZE):
        super(ContrastiveLoss, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, z, q, v):
        

        zq = F.cosine_similarity(z, q , dim = -1)

        zv = F.cosine_similarity(z.view(z.size(0),1,-1), v, dim = -1)

        qv = F.cosine_similarity(q.view(z.size(0),1,-1), v, dim = -1)

        loss_z_anchor = -torch.log(torch.exp(zq/ self.temperature)/ (torch.exp(zq/ self.temperature) + torch.sum(torch.exp(zv/self.temperature), dim = -1)))
        loss_q_anchor = -torch.log(torch.exp(zq/ self.temperature)/ (torch.exp(zq/ self.temperature) + torch.sum(torch.exp(qv/self.temperature), dim = -1)))

        loss = torch.cat([loss_z_anchor, loss_q_anchor], dim = 0)
        
        return loss.mean()