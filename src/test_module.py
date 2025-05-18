import torch
import numpy as np
import math
import numbers
from src.hypotest import multi_checkmean, multi_checkcov, lenth4pvalue, angle4pvalue, ks_test

def attack_metric(predicts, signature):
    L_P = lenth4pvalue(predicts)
    A_P = angle4pvalue(predicts, signature, ifunity=True)

    return L_P, A_P


def test_normality(pred): 
    tested_obs = pred.to('cpu')
    
    # for the mean hypothesis test
    mean_p = multi_checkmean(tested_obs, mean=0.0, confidence = 0.05)

    # for the covariance hypothesis test
    cov_p = multi_checkcov(tested_obs, cov=None, confidence = 0.05)

    if not (np.isfinite(mean_p) and np.isfinite(cov_p)):
        print("The mean_p or cov_p is NAN or INF, check your inputs or the computation")

    norm_p = ks_test(tested_obs)

    return mean_p, cov_p, norm_p

def unsuccess_ratio(dict, confidence, ifcorrection=True, ifprint=True):
    index_dict = {}
    result_dict = {}
    keys_list = list(dict.keys())
    keys_len = len(keys_list)
    for i in range(keys_len):

        for item in dict[keys_list[i]]:
            if not isinstance(item, numbers.Number):
                raise ValueError("List contains non-number items.")
        index_dict[keys_list[i]] = []

    for key in keys_list:
        if ifcorrection:
            steps = len(dict[keys_list[0]])
            new_confidence = confidence/steps
            index_dict[key] = np.array(dict[key])<new_confidence
        else:
            index_dict[key] = np.array(dict[key])<confidence
        index_dict[key] = index_dict[key].reshape(-1, 1)
        result_dict[key] = np.sum(index_dict[key], axis=0)/index_dict[key].shape[0]
    
    if ifprint:
        print(f"percentage of hypothesis test w/ p<{confidence}: ")
        for key in keys_list:
            # keep two decimal places
            print("\t" ,key, ": ", round(result_dict[key][0], 4))
    print('\n')
    
    return result_dict

def checknaninf(input, description):
    if torch.isnan(input).sum() > 0:
        print(f"Existing NAN in the input vector of {description}")

    if torch.isinf(input).sum() > 0:
        print(f"Existing INF in the input vector of {description}")

def multiplying_pvalue(p1, p2, w1=1, w2=1 ):
    p1 = torch.tensor(p1)
    checknaninf(p1, "LENGTH")
    p2 = torch.tensor(p2)
    checknaninf(p2, "ANGLE")
    p = []
    eps = 1e-6
    for i in range(len(p1)):
        if w2/w1 == 1:
            p.append(p1[i]*p2[i]*(1-math.log(p1[i]*p2[i]+eps)))
        else:
            p.append(p1[i]*p2[i]*(p2[i]**(w2/w1-1)/(1-w2/w1)+p1[i]**(w1/w2-1)/(1-w1/w2)))
    
    p = torch.tensor(p)  # convert list back to tensor for the following checks
    checknaninf(p, "COMBINE")
    # check_output = torch.isnan(p) + torch.isinf(p)
    # if check_output.sum() > 0:
    #     print("The p value is NAN or INF")
    return p.numpy()


def test_normality_intraining(dataloader, model, loss_funcs=None, loss_wgt=None, device=None, ifcorrection=True):
    # design: use means and covariance of a batch of traning (or test) dataset to conduct a hypothesis test repectively. 
    # And count the number of times that the null hypothesis is rejected for mean and covariance respectively.
    # Each null hypothesis is that the mean is 0 and the covariance is identity matrix respectively.
    # If the p-value is less than the confidence (usually 0.05), we reject the null hypothesis.
    # If the p-value is greater than the confidence, we accept the null hypothesis, Concrectly, the we don't have significant evidence to reject the null hypothesis.

    num_batches = len(dataloader)
    model.eval()
    steps = 0.0

    mean_p_list = []
    cov_p_list = []
    norm_p_list = []
    val_loss = 0

    with torch.no_grad():
        for imgs, _ in dataloader:
            
            steps = steps + 1.0
            if device:
                imgs = imgs.to(device)
            pred = model(imgs)

            if loss_funcs is not None:
                assert loss_wgt is not None
                loss = 0
                for i in range(len(loss_funcs)):
                    loss = loss + loss_wgt[i]*loss_funcs[i](pred)
                val_loss += loss.item()

            tested_obs = pred.clone().detach().to('cpu')
        
            # for the mean hypothesis test
            mean_p = multi_checkmean(tested_obs, mean=0.0, confidence = 0.05)
            mean_p_list.append(mean_p)

            # for the covariance hypothesis test
            cov_p = multi_checkcov(tested_obs, cov=None, confidence = 0.05)
            cov_p_list.append(cov_p)

            # for the normality hypothesis test
            norm_p = lenth4pvalue(tested_obs)
            norm_p_list.extend(norm_p)

    val_loss /= steps

    if len(cov_p_list) != len(mean_p_list):
        raise ValueError('The length of the two lists for attack-testing are not equal.')
    else:
        steps = len(cov_p_list)
    

    if ifcorrection:
        new_confidence = 0.05/steps
        index4meanp = np.array(mean_p_list)<new_confidence
        index4covp = np.array(cov_p_list)<new_confidence
    else:
        index4meanp = np.array(mean_p_list)<0.05
        index4covp = np.array(cov_p_list)<0.05
    
    index4normal = np.array(norm_p_list)<0.05

    result4meanp = sum(index4meanp)/steps
    result4covp = sum(index4covp)/steps
    result4normal = sum(index4normal)/len(norm_p_list)

    p_list_dict = {'mean_p': mean_p_list, 'cov_p': cov_p_list, 'normal_p': norm_p_list}

    return result4meanp, result4covp, result4normal, val_loss, p_list_dict