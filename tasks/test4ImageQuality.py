import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import math
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)


from src.model import ResidualBlock,ResNet
from src.attack import LinfPGDAttack
from src.loss import AdvWmLoss

from src.test_module import unsuccess_ratio
from src.utils import load_model, set_random_seed
from src.generate_signatures import orthogonal_signatures,alignment_signatures, naive_signatures
from src.dataset import CusDataset
from src.visualization_metrics import compute_metrics as vcm

set_random_seed(302)


def test_ImageQuality(dataloader, loss_fn, model, epsilon=8/255, steps=150, alpha=2/255, device=None):

    model.eval()
    output_size = model.output_size
    batch_size = dataloader.batch_size

    attacker = LinfPGDAttack(model, loss_fn,
                                    epsilon=epsilon, steps=steps, alpha=alpha, random_start=False)

    vis_metrics = {'psnr':[], 'ssim':[], 'rmse':[], 'mae':[]}
    vis_values={}  

    with torch.no_grad():
        for step, (imgs, imgnames) in enumerate( dataloader):
            batches = imgs.size(0)
            imgs = imgs.to(device)

            nature_pred = model(imgs)
            y_signature = alignment_signatures(nature_pred, seed=302)
            # y_signature = orthogonal_signatures(y_signature, seed=302)
            # y_signature = naive_signatures(y_signature, seed=302)
            target = y_signature.clone().detach().requires_grad_(True)
            adv_imgs = attacker.perturb(imgs, target)
            attack_pred = model(adv_imgs)

            vis_values = vcm(imgs, adv_imgs)
            for key in vis_metrics.keys():
                vis_metrics[key].extend(vis_values[key])

    return vis_metrics,

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device} device')

test_batch_size = 1
output_size = 32
load_name = "./stored_models/examplemodel_coco_resnet18.pt"

transform_test = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

transnorm = {'mean':[0.5, 0.5, 0.5], 'std':[0.5, 0.5, 0.5]} 
test_img_folder = '../data/coco2017/images/val2017'
test_data = CusDataset(root_dir=test_img_folder, train=False, transform=transform_test)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=4)
sub_test_data = torch.utils.data.Subset(test_data, np.random.choice(len(test_data), 10, replace=False))
sub_test_dataloader = DataLoader(sub_test_data, batch_size=1, shuffle=False, num_workers=4)

loss_fn = AdvWmLoss().to(device)

wmmodel = ResNet(ResidualBlock, output_size=output_size, transnorm=transnorm)
wmmodel = load_model(wmmodel,load_name, prefix='_orig_mod.')
wmmodel = wmmodel.to(device)

results = test_ImageQuality(sub_test_dataloader, loss_fn, wmmodel, epsilon=6.3*1e-4, steps=100, alpha=0.01, device=device)

print('PSNR:', np.round(np.mean(results[0]['psnr']), 4))
print('SSIM:', np.round(np.mean(results[0]['ssim']), 4))
print('RMSE:', np.round(np.mean(results[0]['rmse']), 4))
print('MAE:', np.round(np.mean(results[0]['mae']), 4))