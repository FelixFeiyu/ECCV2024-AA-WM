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
from src.test_module import  attack_metric, multiplying_pvalue
from src.utils import load_model, set_random_seed
from src.generate_signatures import orthogonal_signatures,alignment_signatures, naive_signatures
from src.dataset import CusDataset

set_random_seed(302)

def test_Security1(dataloader, loss_fn, model, epsilon=8/255, steps=150, alpha=2/255, device=None):
    # Try to generate fake signatures randomly.
    model.eval()
    output_size = model.output_size
    attacker = LinfPGDAttack(model, loss_fn,
                                    epsilon=epsilon, steps=steps, alpha=alpha, random_start=False)

    p_values = []

    batch_size = dataloader.batch_size
    with torch.no_grad():
        for step, (imgs, imgnames) in enumerate( dataloader):
            batches = imgs.size(0)
            imgs = imgs.to(device)

            nature_pred = model(imgs)
            y_signature = alignment_signatures(nature_pred, seed=302).requires_grad_(True)
            fake_signatures = naive_signatures(torch.randn(100, output_size), seed=3023).requires_grad_(True)
            fake_signature = fake_signatures[0].unsqueeze(0).to(device)
            wm_imgs = attacker.perturb(imgs, y_signature)
            wm_pred = model(wm_imgs)

            _, p_angle = attack_metric(wm_pred, fake_signature)
            p_values.extend(p_angle)

    percentage = np.round(np.sum(np.array(p_values)<0.05)/len(p_values), 4)*100
    print(f'Using fake randomly-generated signatures, the percentage of successful steal is {percentage}%')

    return None

def test_Security2(dataloader, loss_fn, model, fake_models, epsilon=8/255, steps=150, alpha=2/255, device=None):
    # Try to using another SKN to generate fake signatures.
    model.eval()
    output_size = model.output_size
    attacker = LinfPGDAttack(model, loss_fn,
                                    epsilon=epsilon, steps=steps, alpha=alpha, random_start=False)

    p_values = []

    batch_size = dataloader.batch_size
    with torch.no_grad():
        for step, (imgs, imgnames) in enumerate( dataloader):
            batches = imgs.size(0)
            imgs = imgs.to(device)

            nature_pred = model(imgs)
            y_signature = alignment_signatures(nature_pred, seed=302).requires_grad_(True)
            wm_imgs = attacker.perturb(imgs, y_signature).clamp(0, 1)
            wm_pred = model(wm_imgs)
            
            fake_model = fake_models[-1].eval().to(device)
            fake_signature = fake_model(wm_imgs).clone().detach().requires_grad_(True)
            _, p_angle = attack_metric(wm_pred, fake_signature)
            p_values.extend(p_angle)

    percentage = np.round(np.sum(np.array(p_values)<0.05)/len(p_values), 4)*100
    print(f'Using fake orthogonalized signatures, the percentage of successful steal is {percentage}%')

    return None

def test_Security3(dataloader, loss_fn, model, fake_models, epsilon=8/255, steps=150, alpha=2/255, device=None):
    # Try to overlay orignal watermark with different fake signatures but same SKN 4 times.
    model.eval()
    output_size = model.output_size
    attacker = LinfPGDAttack(model, loss_fn,
                                    epsilon=epsilon, steps=steps, alpha=alpha, random_start=False)

    p_values = []

    batch_size = dataloader.batch_size
    
    angle_lenth_dict = {'wm':[], 'f1':[], 'f2':[], 'f3':[], 'f4':[]}

    with torch.no_grad():
        for step, (imgs, imgnames) in enumerate( dataloader):
            batches = imgs.size(0)
            imgs = imgs.to(device)

            nature_pred = model(imgs)
            y_signature = alignment_signatures(nature_pred, seed=302).requires_grad_(True)
            wm_imgs = attacker.perturb(imgs, y_signature).clamp(0, 1)
            wm_pred = model(wm_imgs)

            p_length, p_angle = attack_metric(wm_pred, y_signature)
            p = multiplying_pvalue(p_length, p_angle)
            angle_lenth_dict[f'wm'].extend(p)

            for i in range(4):
                fake_signature = naive_signatures(torch.randn(100, output_size), seed=303+i).requires_grad_(True)
                fake_signature = fake_signature[0].unsqueeze(0).to(device)
                wm_imgs = attacker.perturb(wm_imgs, fake_signature)
                wm_pred = model(wm_imgs)
                p_length, p_angle = attack_metric(wm_pred, y_signature)
                p = multiplying_pvalue(p_length, p_angle)
                angle_lenth_dict[f'f{i+1}'].extend(p)
        times = 0
        for key in angle_lenth_dict.keys():
            percentage = np.round(np.sum(np.array(angle_lenth_dict[key])<0.05)/len(angle_lenth_dict[key]), 4)*100
            if key == 'wm':
                print(f'Percentage of successful detection rate of watermark is {percentage}%')
            else:
                print(f'Percentage of detecting original watermark when overwriting with fake signatures {times} times is {percentage}%')
            times+=1
    return None

def test_Security4(dataloader, loss_fn, model, fake_models, epsilon=8/255, steps=150, alpha=2/255, device=None):
    # Try to overlay orignal watermark with different fake signatures and SKN 4 times.
    model.eval()
    output_size = model.output_size
    attacker = LinfPGDAttack(model, loss_fn,
                                    epsilon=epsilon, steps=steps, alpha=alpha, random_start=False)

    p_values = []

    batch_size = dataloader.batch_size
    
    angle_lenth_dict = {'wm':[], 'f1':[], 'f2':[], 'f3':[], 'f4':[]}

    with torch.no_grad():
        for step, (imgs, imgnames) in enumerate( dataloader):
            batches = imgs.size(0)
            imgs = imgs.to(device)

            nature_pred = model(imgs)
            y_signature = alignment_signatures(nature_pred, seed=302).requires_grad_(True)
            attacker.model = model
            wm_imgs = attacker.perturb(imgs, y_signature).clamp(0, 1)
            wm_pred = model(wm_imgs)

            p_length, p_angle = attack_metric(wm_pred, y_signature)
            p = multiplying_pvalue(p_length, p_angle)
            angle_lenth_dict[f'wm'].extend(p)

            for i in range(4):
                fake_signatures = naive_signatures(torch.randn(100, output_size), seed=303+i).requires_grad_(True)
                fake_signature = fake_signatures[0].unsqueeze(0).to(device)
                attacker.model = fake_models[i].eval().to(device)
                wm_imgs = attacker.perturb(wm_imgs, fake_signature)
                wm_pred = model(wm_imgs)
                p_length, p_angle = attack_metric(wm_pred, y_signature)
                p = multiplying_pvalue(p_length, p_angle)
                angle_lenth_dict[f'f{i+1}'].extend(p)
        
        times = 0
        for key in angle_lenth_dict.keys():
            percentage = np.round(np.sum(np.array(angle_lenth_dict[key])<0.05)/len(angle_lenth_dict[key]), 4)*100
            if key == 'wm':
                print(f'Percentage of successful detection rate of watermark is {percentage}%')
            else:
                print(f'Percentage of detecting original watermark when overwriting with fake signatures {times} times is {percentage}%')
            times+=1
    return None


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
wmmodel = wmmodel.eval().to(device)

fake_model_pathes = ['./stored_models/fake_resnet18_1.pt',
                     './stored_models/fake_resnet18_2.pt',
                     './stored_models/fake_resnet18_3.pt',
                     './stored_models/fake_resnet18_4.pt']

fake_models = []
for fake_model_path in fake_model_pathes:
    fake_model = ResNet(ResidualBlock, output_size=output_size, transnorm=transnorm)
    fake_model = load_model(fake_model,fake_model_path, prefix='_orig_mod.')
    fake_models.append(fake_model.eval())

test_Security1(test_dataloader, loss_fn, wmmodel, epsilon=6.3*1e-4, steps=100, alpha=0.01, device=device)
test_Security2(test_dataloader, loss_fn, wmmodel, fake_models, epsilon=6.3*1e-4, steps=100, alpha=0.01, device=device)
test_Security3(test_dataloader, loss_fn, wmmodel, fake_models, epsilon=6.3*1e-4, steps=100, alpha=0.01, device=device)
test_Security4(test_dataloader, loss_fn, wmmodel, fake_models, epsilon=6.3*1e-4, steps=100, alpha=0.01, device=device)
