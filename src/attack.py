import torch
import numpy as np

import src.data_augmentation as da


def mse_clip(x, y, target_mse=1e-4):
    """ 
    Clip x so that MSE(x,y)=target_mse for each image in the batch
    Args:
        x: Image tensor with values between [0,1]
        y: Image tensor with values between [0,1], ex: original image
        target_mse: Target MSE value. when PSNR = 40, MSE=1e-4;  when PSNR=42, MSE=6.31*1e-5`
    Returns:
        Clipped x tensor and MSE values for each image in the batch
    """
    delta = x - y

    # Calculate MSE for each image in the batch
    mse_per_image = torch.mean(delta**2, dim=[1,2,3])

    # Find images for which MSE is not equal to target_mse and adjust them
    scaling_factors =torch.clamp( torch.sqrt(target_mse / mse_per_image), 0, 1)
    
    return y + (delta.transpose(0,-1) * scaling_factors).transpose(0,-1)


class LinfPGDAttack():
    def __init__(self, model, loss_fn, epsilon=8/255, steps=20, alpha=2/255, random_start=False):
        self.model = model.eval()
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.rand = random_start

    def perturb(self, nature_imgs, targets, angle=None, length=None):

        if self.rand:
            perturbation = torch.empty_like(nature_imgs).uniform_(-self.epsilon, self.epsilon)
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            imgs = nature_imgs.detach() + perturbation
            imgs = torch.clamp(imgs, 0, 1)
        else:
            imgs = nature_imgs.clone().detach()

        
        for _ in range(self.steps):
            imgs.requires_grad_()
            
            with torch.enable_grad():
                data_aug_f = da.All()
                aug_params = data_aug_f.sample_params(nature_imgs[0])
                aug_imgs = data_aug_f(imgs, aug_params)
                outputs = self.model(aug_imgs)

                loss_la = self.loss_fn(outputs, targets)
                # loss_l2 = torch.tensor(-1)*torch.sum((imgs-nature_imgs)**2) 
                # loss = loss_la+loss_l2
                loss = loss_la

                grad = torch.autograd.grad(loss, [imgs])[0]
                imgs = imgs.detach() + self.alpha * grad

                imgs = mse_clip(imgs, nature_imgs, self.epsilon) #0.000631
                imgs = torch.clamp(imgs, 0, 1) 
        
        return imgs



    
