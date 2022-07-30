from torch.utils.data import DataLoader
from learner import Learner
from loss import *
from dataset import *
import os
from sklearn import metrics
from tqdm import tqdm
import config
from contrastivehead import ContrastiveHead
from contrastiveloss import ContrastiveLoss
from milmodel import MILModel


#Setting the seed
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(config.SEED)

normal_train_dataset = Normal_Loader(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)

anomaly_train_dataset = Anomaly_Loader(is_train=1)
anomaly_test_dataset = Anomaly_Loader(is_train=0)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=True)

anchor_train_loader = DataLoader(anomaly_train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Learner().to(device)
model.load_state_dict(torch.load(os.path.join(config.SAVE_DIR, "milmodel1.pth")))
model.eval()
contrastive_head = ContrastiveHead()
contrastive_head.train()
optimizer = torch.optim.Adagrad(contrastive_head.parameters(), lr= config.LR_CON, weight_decay=0.0010000000474974513)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = ContrastiveLoss(temperature = config.TEMPERTAURE, batch_size = config.TRAIN_BATCH_SIZE)

def train(epoch, contrastive_head, model):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    # loader = tqdm(zip(normal_train_loader, anomaly_train_loader), total = min(len(normal_train_loader), len(anomaly_train_loader)))
    for batch_idx, (normal_inputs, anomaly_inputs, anchor_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader, anchor_train_loader)):


        z_scores = model(anchor_inputs)
        q_scores = model(anomaly_inputs)
        z_idxs = z_scores.argmax(1).view(-1)
        q_idxs = q_scores.argmax(1).view(-1)

        z = contrastive_head(anchor_inputs)
        q = contrastive_head(anomaly_inputs)
        v = contrastive_head(normal_inputs)


        anchors = []
        positive = []
        for i in range(z_scores.size(0)):
            anchors.append(z[i,z_idxs[i],:].view(1,-1))
            positive.append(q[i,q_idxs[i],:].view(1,-1))
        z = torch.cat(anchors, dim = 0)
        q = torch.cat(positive, dim = 0)

        loss = criterion(z, q, v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('loss = {}'.format(train_loss/len(normal_train_loader)))
    scheduler.step()

    return train_loss/len(normal_train_loader)

best_loss = np.inf
for epoch in range(0, config.EPOCHS_CONTRASTIVE):


    epoch_loss = train(epoch, contrastive_head, model)
    if epoch_loss < best_loss:
        model_save_path = os.path.join(config.SAVE_DIR, "contrastivehead.pth")
        torch.save(contrastive_head.state_dict(), model_save_path)
        best_loss = epoch_loss

    print(best_loss)
    

