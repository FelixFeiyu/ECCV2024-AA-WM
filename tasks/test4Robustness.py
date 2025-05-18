import os
import shutil
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.model import ResNet, ResidualBlock
from src.loss import AdvWmLoss
from src.attack import LinfPGDAttack
from src.utils import load_model, set_random_seed
from src.generate_signatures import naive_signatures, orthogonal_signatures, alignment_signatures
from src.dataset import CusDataset
from src.test_module import attack_metric, multiplying_pvalue, unsuccess_ratio
from src.perturbations import apply_perturbations


def save_images(images, filenames, save_dir):
    for img, fname in zip(images, filenames):
        img_path = os.path.join(save_dir, fname)
        img.save(img_path)

set_random_seed(302)

original_folder = 'outputs/tmp_original_images'
adversarial_folder = 'outputs/robustness/tmp_adv_images'
difference_folder = 'outputs/robustness/tmp_diff_images'
y_signaures_path =  'outputs/robustness/tmp_y_signatures.pkl'

for folder in [original_folder, adversarial_folder, difference_folder]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

transform_test = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

test_img_folder = '../data/coco2017/images/val2017'
# test_img_folder = '../data/imagenet2012/val'
test_data = CusDataset(dataset_name='COCO', root_dir=test_img_folder, train=False, transform=transform_test, if_save=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
output_size = 32

# load model
load_name = "./stored_models/examplemodel_coco_resnet18.pt"
transnorm = {'mean':[0.5, 0.5, 0.5], 'std':[0.5, 0.5, 0.5]} 
wmmodel = ResNet(ResidualBlock, output_size=output_size, transnorm=transnorm)
wmmodel = load_model(wmmodel, load_name, prefix='_orig_mod.')
wmmodel = wmmodel.to(device)
wmmodel.eval()


loss_fn = AdvWmLoss().to(device)
attacker = LinfPGDAttack(wmmodel, loss_fn,
                        epsilon=0.00063, steps=100, alpha=0.01, random_start=False)

y_signatures_dict = {}
with torch.no_grad():
    for imgs, filenames in test_dataloader:
        imgs = imgs.to(device)
        nature_pred = wmmodel(imgs)
        y_signature = alignment_signatures(nature_pred, seed=302)

        target = y_signature.clone().detach().requires_grad_(True)
        adv_imgs = attacker.perturb(imgs, target)
        adv_pred = wmmodel(adv_imgs)
        p_lenth, p_angle = attack_metric(adv_pred, y_signature)
        p = multiplying_pvalue(p_lenth, p_angle)
      
        adv_imgs = adv_imgs.clamp(0, 1)
        adv_imgs_pil = [transforms.ToPILImage()(img) for img in adv_imgs.cpu()]
        filenames = [filename.replace('.JPEG', '.png') for filename in filenames]
        save_images(adv_imgs_pil, filenames, adversarial_folder)

        nature_imgs_pil = [transforms.ToPILImage()(img) for img in imgs.cpu()]
        save_images(nature_imgs_pil, filenames, original_folder)

        y_signatures_dict[filenames[0].split('.')[0]] = y_signature.cpu()
        with open(y_signaures_path, 'wb') as f:
            pickle.dump(y_signatures_dict, f)


print('watermarking done!!!')
batch_size = 200
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_advimg_folder = adversarial_folder
test_advdata=CusDataset(root_dir=test_advimg_folder, train=False, transform=transform_test, if_save=True)
test_advdataloader = DataLoader(test_advdata, batch_size=batch_size, shuffle=False)

test_orgimg_folder = original_folder
test_orgdata=CusDataset(root_dir=test_orgimg_folder, train=False, transform=transform_test, if_save=True)
test_orgdataloader = DataLoader(test_orgdata, batch_size=batch_size, shuffle=False)

with open(y_signaures_path, 'rb') as f:
    y_signatures_dict = pickle.load(f)

perturbations = {
    "gaussian_blur": {'kernel_size': [1+2*i for i in range(1,10)]},
    'gaussian_noise': {'sigma': [0.02*i for i in range(1,8)]},
    'rotation': {'degrees': [i for i in range(0,45,5)]},
    'cropping_ratio': {'size_ratio': [0.1*i for i in range(1,11)]},
    'JPEG': {'quality': [10*i for i in range(1,11)]},
}

perturb_results_dict = {}
for ptb_type, ptb_hyp_dict in perturbations.items():
    for ptb_hyp_name, ptb_hyp_values in ptb_hyp_dict.items():

        for hyp_value in ptb_hyp_values:
            hyp_value = round(hyp_value, 2)
            perturb_results_dict[f'{ptb_type}_{ptb_hyp_name}_{hyp_value}'] = {'length':[], 'angle':[], 'mtp':[]}

            print(ptb_type, ptb_hyp_name, hyp_value)
            
            with torch.no_grad():
                for ii, (adv_zip, org_zip) in enumerate(zip(test_advdataloader, test_orgdataloader)):
                    adv_imgs, adv_filenames = adv_zip
                    org_imgs, org_filenames = org_zip
                    adv_imgs = adv_imgs.to(device)
                    org_imgs = org_imgs.to(device)
                    y_signatures = []
                    for i in range(len(adv_filenames)):
                        y_signature = y_signatures_dict[adv_filenames[i].split('.')[0]].to(device)
                        y_signatures.append(y_signature)
                    y_signatures = torch.cat(y_signatures, dim=0)

                    ptb_imgs = apply_perturbations([adv_imgs, org_imgs], {ptb_type: {ptb_hyp_name: hyp_value}})
                    ptb_imgs = ptb_imgs.clamp(0, 1)
                    ptb_preds = wmmodel(ptb_imgs)
            

                    p_lenth, p_angle = attack_metric(ptb_preds, y_signatures)
                    p = multiplying_pvalue(p_lenth, p_angle)
                    perturb_results_dict[f'{ptb_type}_{ptb_hyp_name}_{hyp_value}']['length'].extend(p_lenth)
                    perturb_results_dict[f'{ptb_type}_{ptb_hyp_name}_{hyp_value}']['angle'].extend(p_angle)
                    perturb_results_dict[f'{ptb_type}_{ptb_hyp_name}_{hyp_value}']['mtp'].extend(p)

print('starting to store the pickle results')

outfile_name = f'./outputs/robustness/pvalues_after_ptb.pkl'
with open(outfile_name, 'wb') as f:
    pickle.dump(perturb_results_dict, f)
    print(f'stored the pickle results in {outfile_name}')

# print('starting to store the csv results')
# outfile_name = f'./outputs/robustness/pvalues_after_ptb.csv'
# max_values = 5000
# columns = ['ptb_config', 'metric_type'] + ['imgs_' + str(i) for i in range(1, max_values+1)]
# rows_list = []
# for key, metrics in perturb_results_dict.items():
#     ptb_config = key
#     for metric_type, values in metrics.items():
#         row = [ptb_config, metric_type] + [value for value in values]
#         rows_list.append(row)
# df = pd.DataFrame(rows_list, columns=columns)
# df.to_csv(outfile_name, index=False)
# print(f'stored the csv results in {outfile_name}')


print('Percents of detecting the correct watermark (p-value < 0.05):')
for kptb_config, metrics in perturb_results_dict.items():

    unsucc_ratio = unsuccess_ratio(metrics, 0.05,False, False)
    for metric_type, values in metrics.items():
        print(kptb_config, metric_type, unsucc_ratio[metric_type][0]*100)


