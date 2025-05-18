from random import randint
from torchvision.transforms.functional import rotate, crop, pad, resize, InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms import ToPILImage, ToTensor, RandomCrop, GaussianBlur, ColorJitter, CenterCrop
from PIL import Image, ImageFilter
import torch
import numbers
from collections.abc import Sequence
import io

import torch
import torch.nn as nn
import math

class Gaussian(nn.Module):
    def __init__(self, num_features, sigma):
        super(Gaussian, self).__init__()
        self.radius = math.ceil(2 * sigma)
        self.padding = nn.ReplicationPad2d(self.radius)
        self.conv = GaussianConvolution(num_features, sigma, self.radius)
    
    def forward(self, x):
        x = self.padding(x)
        return self.conv(x)


class GaussianConvolution(nn.Module):
    def __init__(self, num_features, sigma, radius):
        super(GaussianConvolution, self).__init__()
        self.num_features = num_features
        self.sigma = sigma
        self.radius = radius
        self.conv = nn.Conv2d(num_features, num_features, 2 * radius + 1, stride=1, padding=0, bias=False)
        self.conv.weight.data = self.gaussianMatrix()
    
    def gaussianMatrix(self):
        weight = torch.zeros(self.num_features, self.num_features, 2 * self.radius + 1, 2 * self.radius + 1)
        for f in range(self.num_features):
            for x in range(2 * self.radius + 1):
                for y in range(2 * self.radius + 1):
                    weight[f, f, x, y] = math.exp(- (x * x + y * y) / (2 * self.sigma * self.sigma))
        weight *= 1 / weight[0, 0].sum()
        return weight

    def forward(self, x):
        return self.conv(x)
    
    def __str__(self):
        return super().__str__() + f' <gaussian: sigma: {self.sigma}, radius: {self.radius}> '


def get_perturbations_info(ptb_type):
    '''
    defining this function is to get the perturbation type and parameters names, convenient fo the tittle of ploting.
    '''
    perturbations_names = {
        "gaussian_noise": "Gaussian Noise",
        "gaussian_blur": "Gaussian Blur",
        "rotation": "Rotation",
        "cropping_ratio": "Cropping + Resizing",
        "JPEG": "JPEG Compression",
    }
    params_names = {        
        "gaussian_noise": "Sigma",
        "gaussian_blur": "Kernel Size",
        "rotation": "Degrees",
        "cropping_ratio": "Size Ratio",
        "JPEG": "Quality",
    }
    return perturbations_names[ptb_type], params_names[ptb_type]


def apply_perturbations(imgs, perturbations):
    '''
    apply_perturbations(img, perturbations)
    Applies perturbations to an input image based on the provided dictionary.

    Args:
        img (torch.Tensor or list): The input image tensor to perturb of shape (B, C, H, W).
        or a list of perturbed images and original images
        perturbations (dict): A dictionary containing the types and parameters of perturbations to apply.

    Returns:
        torch.Tensor: The perturbed image tensor of shape (B, C, H, W).
    '''

    def rotate_image(images, angle):
        return rotate(images, angle, interpolation=InterpolationMode('bilinear'), fill=1)
        #return rotate(images, angle,)
        #return torch.stack([rotate(img, angle, fill=0) for img in image])

             
    def add_gaussian_noise(images, sigma):
        noise = torch.randn_like(images) * sigma
        # return torch.clamp(image + noise, 0, 1)
        return images + noise
    
    def gaussian_blur(images, kernel_size):
        images = GaussianBlur(kernel_size)(images)
        return images


    def crop_ratio(images, **kwargs):
        """
        Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
        size_ratio, or height_ratio_range and width_ratio_range
        size_ratio: the ratio of the cropped image to the original image, float or tuple of floats
        height_ratio: the ratio of the cropped image to the original image in the vertical direction, 
                      float or tuple of floats
        width_ratio: the ratio of the cropped image to the original image in the horizontal direction, 
                     float or tuple of floats
        """

        # check the lenth of the **kwargs
        if len(kwargs) == 1:
            height_ratio = math.sqrt(kwargs['size_ratio'])
            width_ratio = math.sqrt(kwargs['size_ratio'])

        elif len(kwargs) == 2:
            height_ratio = kwargs['height_ratio']
            width_ratio = kwargs['width_ratio']
            height_ratio=torch.empty(1).uniform_(height_ratio[0], height_ratio[1]).item()
            width_ratio=torch.empty(1).uniform_(width_ratio[0], width_ratio[1]).item()

        else:
            raise ValueError("Too many arguments for crop_ratio.")

        # if isinstance(height_ratio, numbers.Number):
        #     height_ratio = (height_ratio, height_ratio)
        # if isinstance(width_ratio, numbers.Number):
        #     width_ratio = (width_ratio, width_ratio)
        
        image_height, image_width = images.shape[2:]
        crop_height = int(height_ratio * image_height)
        crop_width = int(width_ratio* image_width)

        random_crop = CenterCrop((crop_height, crop_width))


        return resize(random_crop(images), (image_height, image_width))
    

    def JPEG(images, quality):
        assert 0 <= quality <= 100, "'quality' must be a value in the range [0, 100]"
        #assert isinstance(images, Image.Image), "Expected type PIL.Image.Image for variable 'image'"

        batch_aug_image = []
        for image in images:
            image = ToPILImage()(image)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            aug_image = Image.open(buffer)
            aug_image = ToTensor()(aug_image)
            batch_aug_image.append(aug_image.unsqueeze(0))

        batch_aug_image = torch.cat(batch_aug_image,dim = 0).to(images.device)
        return batch_aug_image
    

    
    if isinstance(imgs, list):
        perturbed_imgs= imgs[0]
        original_imgs = imgs[1]
    else:
        perturbed_imgs = imgs
    
    for perturbation, params in perturbations.items():
        if perturbation == 'identity':
            perturbed_imgs = perturbed_imgs

        if perturbation == "gaussian_noise":
            perturbed_imgs = add_gaussian_noise(perturbed_imgs, params["sigma"])
        
        elif perturbation == "gaussian_blur":
            perturbed_imgs = gaussian_blur(perturbed_imgs, params["kernel_size"])

        elif perturbation == "rotation":
            perturbed_imgs = rotate_image(perturbed_imgs, params["degrees"])
        
        elif perturbation == "cropping_ratio":
            perturbed_imgs = crop_ratio(perturbed_imgs, **params)
        
        elif perturbation == "JPEG":
            perturbed_imgs = JPEG(perturbed_imgs, params['quality'])

            
    return perturbed_imgs
    