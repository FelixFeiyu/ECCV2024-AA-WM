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
from src.test_module import unsuccess_ratio, attack_metric, multiplying_pvalue
from src.utils import load_model, set_random_seed, normalizedobs
from src.generate_signatures import orthogonal_signatures,alignment_signatures, naive_signatures
from src.dataset import CusDataset


set_random_seed(302)


def test_DetectionAccuracy(dataloader, loss_fn, model, epsilon=8/255, steps=150, alpha=2/255, device=None):
    model.eval()
    output_size = model.output_size
    attacker = LinfPGDAttack(model, loss_fn,
                                    epsilon=epsilon, steps=steps, alpha=alpha, random_start=False)

    angle_lenth_dict = {'wm_lenth_p':[], 'wm_angle_p':[], 'wm_mtp_p':[],}

    batch_size = dataloader.batch_size
    with torch.no_grad():
        for step, (imgs, imgnames) in enumerate( dataloader):
            batches = imgs.size(0)
            imgs = imgs.to(device)

            nature_pred = model(imgs)
            y_signature = alignment_signatures(nature_pred, seed=302)
            # y_signature = orthogonal_signatures(y_signature, seed=302)
            # y_signature = naive_signatures(y_signature, seed=302)
            target = y_signature.clone().detach().requires_grad_(True)
            wm_imgs = attacker.perturb(imgs, target)
            wm_pred = model(wm_imgs)

            p_lenth, p_angle = attack_metric(wm_pred, target)
            angle_lenth_dict[f'wm_lenth_p'].extend(p_lenth)
            angle_lenth_dict[f'wm_angle_p'].extend(p_angle)
            p = multiplying_pvalue(p_lenth, p_angle)
            angle_lenth_dict[f'wm_mtp_p'].extend(p)

    return angle_lenth_dict



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

loss_fn = AdvWmLoss().to(device)

wmmodel = ResNet(ResidualBlock, output_size=output_size, transnorm=transnorm)
wmmodel = load_model(wmmodel,load_name, prefix='_orig_mod.')
wmmodel = wmmodel.to(device)

results = test_DetectionAccuracy(test_dataloader, loss_fn, wmmodel, epsilon=6.3*1e-4, steps=100, alpha=0.01, device=device)

percentage = unsuccess_ratio(results, confidence=0.05, ifcorrection=False, ifprint=False)
print('The percentage of output macthing SKS is:', percentage['wm_lenth_p'][0]*100, '%')
print('The percentage of output macthing SKN is:', percentage['wm_angle_p'][0]*100, '%')
print('The percentage of output macthing SKS and SKN is:', percentage['wm_mtp_p'][0]*100, '%')