import torch
from scipy.stats import f, chi2, norm, kstest
import numpy as np
import scipy.linalg as la
import math


def multi_checkmean(observations, mean, confidence=0.05, ifprint=False): 
    # oberservations is a matrix, columns are features and rows are samples, mean is a vector; 
    # mean [1 x feature_dim]
    # calculate the Hotelling T2 statistics
    n = observations.shape[0]
    p = observations.shape[1]

    x_bar = torch.mean(observations, dim=0, keepdim=True)
    T2 = n*(x_bar - mean)@(x_bar - mean).T
    test_statistic = T2

    p_value = chi2.sf(test_statistic, p)
    #p_value = torch.from_numpy(p_value)
    
    if ifprint:
        if p_value<confidence:
            print(f"under the confidnce {confidence}, the mean of sample is not equal to the one of populations. (p={p_value.item()})")
        else:
            print(f"under the confidnce {confidence}, the mean of sample is equal to the one of populations. (p={p_value.item()})")

    return p_value.item()



def multi_checkcov(observations, cov=None, confidence=0.05, ifprint=False): 
    # oberservations is a matrix, columns are features and rows are samples, cov is a matrix;
    # cov [feature_dim x feature_dim]
    # calculate the Hotelling Kai-square (X2) statistics
    n = observations.shape[0]
    p = observations.shape[1]

    S = torch.cov(observations.T)*(n-1)

    term1 = torch.trace(S)
    term2 = n*torch.logdet(S/torch.tensor(n))
    term3 = torch.tensor(n*p)
    statistic = term1 - term2 - term3

    p_value = chi2.sf(statistic, p*(p+1)/2)
    if ifprint:
        if p_value<confidence:
            print(f"under the confidnce {confidence}, the cov of sample is not equal to the one of populations. (p={p_value.item()})")
        else:
            print(f"under the confidnce {confidence}, the cov of sample is equal to the one of populations. (p={p_value.item()})")

    return p_value.item()

    

def ks_test(input):
    # null hypothesis: the data is from a distribution
    # input: a tensor of size (d,n), d is the number of variables, n is the number of samples, at least n=1000
    # if print: print the mean
    # output: p-value of ks test
    # used for testing if the data is from a normal distribution
    x = input.numpy()
    p_values = []
    for i in range(x.shape[0]):
        p_values.append(kstest((x[i,:]-np.mean(x[i,:]))/np.std(x[i,:]), 'norm')[1]) 

    return p_values


def lenth4pvalue(observations):
    feature_dim = observations.shape[1]
    lenth = torch.norm(observations, dim=1)**2
    p_value = chi2.sf(lenth.cpu(), feature_dim)
    return list(p_value) 


def doublefactorial(n):
    if n % 2 == 0:
        results = (n / 2) * torch.log(torch.tensor(2.0, dtype=torch.float64)) + torch.lgamma(n / 2 + 1)
    else:
        k = (n + 1) / 2
        results = torch.lgamma(2 * k + 1) - k * torch.log(torch.tensor(2.0, dtype=torch.float64)) - torch.lgamma(k + 1)
    return results


def angle4pvalue(observations, watermarks, ifunity=True):
    EPSILON = 1e-10

    if ifunity:
        observations = observations / torch.norm(observations, dim=-1, keepdim=True)
        watermarks = watermarks / torch.norm(watermarks, dim=-1, keepdim=True)

    costheta = torch.sum(observations * watermarks, dim=-1)
    costheta = torch.clamp(costheta, -1, 1)

    theta = torch.acos(costheta)

    n = observations.shape[1]
    sumsin = 0.0
    for k in torch.arange(1, (n - 2) / 2 + 1, 1):
        c_n = torch.exp(doublefactorial(2 * k - 2) - doublefactorial(2 * k - 1))
        sin_theta = torch.clamp(torch.sin(theta), -1, 1)
        sumsin += c_n * (sin_theta ** (2 * k - 1))

    p_value = (theta - costheta * sumsin) / math.pi
    p_value = torch.clamp(p_value, 0, 1)

    return p_value.cpu().tolist()









