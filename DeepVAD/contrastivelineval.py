from torch.utils.data import DataLoader
from learner import Learner
from loss import *
from dataset import *
import os
from sklearn import metrics
from tqdm import tqdm
import config
from finalhead import FinalHead
from contrastivehead import ContrastiveHead


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

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

contrastive_head = ContrastiveHead()
contrastive_head.load_state_dict(torch.load(os.path.join(config.SAVE_DIR,"contrastivehead.pth")))
contrastive_head.eval()

final_head = FinalHead()
optimizer = torch.optim.Adagrad(final_head.parameters(), lr= config.CT_LR, weight_decay=config.WD)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = MIL


def train(epoch, final_head):
    print('\nEpoch: %d' % epoch)
    final_head.train()
    train_loss = 0
    correct = 0
    total = 0
    loader = tqdm(zip(normal_train_loader, anomaly_train_loader), total = min(len(normal_train_loader), len(anomaly_train_loader)))
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(loader):

        with torch.no_grad():
            anomaly_inputs = contrastive_head(anomaly_inputs)
            normal_inputs = contrastive_head(normal_inputs)

        
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = final_head(inputs)
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('loss = {}'.format(train_loss/len(normal_train_loader)))
    scheduler.step()

def test(epoch, final_head):
    final_head.eval()
    auc = 0
    all_gt_list = []
    all_score_list = []
    with torch.no_grad():
        for i, (data) in enumerate(tqdm(anomaly_test_loader, total = len(anomaly_test_loader))):
            inputs, gts, frames = data
            inputs = contrastive_head(inputs)
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device(device))
            score = final_head(inputs)
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.linspace(0, frames[0], 33)

            for j in range(32):
                score_list[int(step[j]):(int(step[j+1]))] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1


            all_gt_list.extend(gt_list)
            all_score_list.extend(score_list)

        for i, (data2) in enumerate(tqdm(normal_test_loader, total = len(normal_test_loader))):
            inputs2, gts2, frames2 = data2
            inputs2 = contrastive_head(inputs2)
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device(device))
            score2 = final_head(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.linspace(0, frames2[0], 33)
            for kk in range(32):
                score_list2[int(step2[kk]):(int(step2[kk+1]))] = score2[kk]
            gt_list2 = np.zeros(frames2[0])


            all_gt_list.extend(gt_list2)
            all_score_list.extend(score_list2)
        

    auc = metrics.roc_auc_score(all_gt_list, all_score_list)

    return auc
        
best_auc = 0.5
for epoch in range(0, config.EPOCHS_MIL):
    train(epoch, final_head)
    epoch_auc = test(epoch, final_head)
    if epoch_auc > best_auc:
        model_save_path = os.path.join(config.SAVE_DIR, "finalhead.pth")
        torch.save(final_head.state_dict(), model_save_path)
        best_auc = epoch_auc
    print(f"The AUC SCORE is {epoch_auc}")

print(f"THE BEST AUC SCORE is {best_auc}")

