# ECCV2024-AA-WM
Code for ECCV2024 paper "A Secure Image Watermarking Framework with Statistical Guarantees via Adversarial Attacks on Secret Key Networks"

# A Secure Image Watermarking Framework with Statistical Guarantees via Adversarial Attacks on Secret Key Networks


## Abstract

> **A Secure Image Watermarking Framework with Statistical Guarantees via Adversarial Attacks on Secret Key Networks**<br>
Feiyu CHEN, Wei LIN, Ziquan LIU, Antoni B. CHAN<br>
> **Abstract**: Imperceptible watermarks are essential in safeguarding the content authenticity and the rights of creators in imagery. Recently, several leading approaches, notably zero-bit watermarking, have demonstrated impressive imperceptibility and robustness in image watermarking. However, these methods have security weaknesses, e.g., the risk of counterfeiting and the ease of erasing an existing watermark with another watermark, while also lacking a statistical guarantee regarding the detection performance. To address this issue, we propose a novel framework to train a secret key network (SKN), which serves as a non-duplicable safeguard for securing the embedded watermark. The SKN is trained so that natural images' output obeys a standard multi-variate normal distribution. To embed a watermark, we apply an adversarial attack (a modified PGD attack) on the image such that the SKN produces a secret key signature (SKS) with a longer length. We then derive two hypothesis tests to detect the presence of the watermark in an image via the SKN response magnitude and the SKS angle, which offer a statistical guarantee of the false positive rate. Our extensive empirical study demonstrates that our framework maintains robustness comparable to existing methods and excels in security and imperceptibility.

 <p align="center">
  <img src="./images/model_arch.png" width="550"/>
</p>

> Our secret-key watermarking framework. (a) the SKN is trained so that its output follows a standard multi-variate normal (SMVN) distribution given an input image distribution; (b) given an  image, the watermark is generated using adversarial attack that makes the SKN output the desired secret key signature (SKS) with extended length; (c) the signature is recovered by applying the SKN to the image, and the watermark is detected using hypothesis tests derived from the assumed SMVN distribution of the SKN.

##  Install

Please install the necessary dependencies:

```bash
pip install -r requirements.txt
```


## Train

You can download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/17z1uP2LLSWjKlFBZ6a0Und3FzgYY60LK?usp=drive_link).  
Before doing so, please create a new folder named `stored_models` and place the downloaded models into it.

Alternatively, you can train your own SKN model by running `train.py` in the `\tasks` directory with different random seeds.

> Make sure to update the dataset path before training. For example, if you're using the MS-COCO dataset:

```python
train_img_folder = './data/coco2017/images/train2017'
```

## Test

* Evaluate the SKN's Normality by running `test4Normality.py` to assess its normality. 

* Evaluate watermarking properties—imperceptibility, strength, robustness, and security—using the following scripts:
  * `test4ImageQuality.py` for imperceptibility
  * `test4Normality.py` for strength
  * `test4Robustness.py` for robustness
  * `test4Security.py` for security

All results will be saved in the `outputs` directory.

**Note:** Before running each script, make sure to update the dataset path to match your local environment.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{
chen2024secimgwm
title={A Secure Image Watermarking Framework with Statistical Guarantees via Adversarial Attacks on Secret Key Networks}
author={Feiyu Chen and Wei Lin and Ziquan Liu and Antoni B. Chan},
booktitle={18th European Conference on Computer Vision},
year={2024}
}
```

## Contact

**Please feel free to get in touch at**: `feiyuchen.cs@gmail.com`



