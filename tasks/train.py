import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.loss import MultiWassersteinLoss, VarianceLoss
from src.test_module import test_normality_intraining
from src.utils import  set_random_seed, EarlyStopping
from src.dataset import CusDataset
from src.model import ResNet, ResidualBlock

global output_size


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

experiment_name = 'resnet18-ouput32-coco'
batch_size = 256  
test_batch_size = 100  
epochs = 15 
output_size = 32
lr = 0.001
loss_w = [1.0, 2.5]
milestones = [5] 
load_path = None 
mode_VL = 'L1' # variance loss mode: L1, L2, ReLU. default L1
save_path = os.path.join('./output', experiment_name)

random_seed = 302
set_random_seed(random_seed)

transform_train = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomCrop(128, padding=0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
])

transform_test = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomCrop(128, padding=0),
    transforms.ToTensor(),
])

train_img_folder = '../data/coco2017/images/train2017'
train_data = CusDataset(root_dir=train_img_folder, train=True, transform=transform_train)

test_img_folder = '../data/coco2017/images/val2017'
test_data = CusDataset(root_dir=test_img_folder, train=False, transform=transform_test)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)


transnorm = {'mean':[0.5, 0.5, 0.5], 'std':[0.5, 0.5, 0.5]}
#transnorm = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
wmmodel = ResNet(ResidualBlock, output_size=output_size, transnorm=transnorm).to(device)


if load_path is not None:
    wmmodel.load_state_dict(torch.load(load_path))
    print('model loaded')

wmmodel = wmmodel.to(device)

loss_fn1 = MultiWassersteinLoss().to(device)
loss_fn2 = VarianceLoss(mode=mode_VL).to(device)

optimizer = torch.optim.Adam(wmmodel.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)
 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    wmmodel.train()
    for batch, (imgs, _) in enumerate(train_dataloader): 
        batchsize = imgs.shape[0] 
        if batchsize != batch_size:
            break
        if device:
            imgs = imgs.to(device)

        pred = wmmodel(imgs)

        w = torch.tensor(loss_w).to(device)
        loss1 = loss_fn1(pred)
        loss2 = loss_fn2(pred)
        loss = w[0]*loss1 + w[1]*loss2 

        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()
        
        if batch % 50 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f} step: [{batch * len(imgs):>5d}/{size:>5d}]")

    print('In the test stage, the hypethersis test results:')
    result4meanp, result4covp, result4norm, val_loss, p_values_dict= \
        test_normality_intraining(test_dataloader, wmmodel, 
                                  loss_funcs=[loss_fn1, loss_fn2], loss_wgt=w,
                                  device=device)
    print('Percentage of mean vector being zero vector: {}'.format(1-result4meanp)) # p<0.05 means the output is not zero vector
    print('Percentage of covariance matrix being identity matrix: {}'.format(1-result4covp)) # p<0.05 means the output is not identity matrix
    print('Percentage of normal distribution: {}'.format(1-result4norm)) # p<0.05 means the output is not normal distribution
    scheduler.step()

    early_stopping(val_loss, wmmodel)
    if early_stopping.early_stop:
        print("Early stopping")
        break

final_stored_modelname = os.path.join(save_path, 'final_model.pt')
torch.save(wmmodel.state_dict(), final_stored_modelname)
print('model saved to {}'.format(save_path))
print("Done!")


