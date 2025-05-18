import torch
import numpy as np
import random 
import math
import os


def set_random_seed(seed=0):
    """Sets the random seed for reproducibility across various libraries."""
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 2)
    
    np.random.seed(seed + 3)
    random.seed(seed + 4)


def load_model(model, load_name, prefix='_orig_mod.'):
    pretrained_dict = torch.load(load_name)
    model_dict = model.state_dict()

    if any(key.startswith(prefix) for key in pretrained_dict.keys()):
        pretrained_dict = {k.replace(prefix, ''): v for k, v in pretrained_dict.items() if k.startswith(prefix)}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Load model successfully')

    return model


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0       
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:   # give a better score
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_name = os.path.join(self.path, 'best_model.pt')
        torch.save(model.state_dict(), save_name)
        self.val_loss_min = val_loss


class InputNormalize(torch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_mean = torch.tensor(new_mean).float()
        new_std = torch.tensor(new_std).float()
        new_std = new_std[None, ..., None, None]
        new_mean = new_mean[None, ..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


class InputTransNormalize(torch.nn.Module):
    '''
    A module (custom layer) for trans-normalizing the input back to its original
    mean and standard deviation (user-specified).
    '''
    def __init__(self, orig_mean, orig_std):
        super(InputTransNormalize, self).__init__()
        orig_mean = torch.tensor(orig_mean).float()
        orig_std = torch.tensor(orig_std).float()
        orig_std = orig_std[None, ..., None, None]
        orig_mean = orig_mean[None, ..., None, None]

        self.register_buffer("orig_mean", orig_mean)
        self.register_buffer("orig_std", orig_std)

    def forward(self, x):
        x_trans_normalized = (x * self.orig_std) + self.orig_mean
        x_trans_normalized = torch.clamp(x_trans_normalized, 0, 1)
        return x_trans_normalized


def normalizedobs(observations):
    
    # this fucntion is to make every row vector of an observation matrix follows the N(0, I) distributions
    batchize = observations.shape[0]
    om = torch.mean(observations, dim=0, keepdim=True).repeat(batchize, 1)
    oc = torch.cov(observations.T)
    eps = torch.tensor(1e-15).to(observations.device)
    eigenvalue, eigenvector = torch.linalg.eig(oc)
    cov_root = torch.matmul(eigenvector, torch.matmul(torch.diag(torch.sqrt(eigenvalue+eps)), eigenvector.T)).real
    observations = (observations-om).mm(torch.inverse(cov_root))

    return observations

def check_output(output):
    if output is None:
        raise ValueError("A_OUTPUT is None.")
    
    if isinstance(output, (int, float)):
        if math.isinf(output):
            raise ValueError("A_OUTPUT is an infinite number.")
    elif isinstance(output, (list, tuple)):
        for value in output:
            if math.isinf(value):
                raise ValueError("A_OUTPUT contains an infinite number.")
    else:
        raise TypeError("A_OUTPUT must be a number, list, or tuple.")


def compute_similarity(pred, gt, mode='L1'):
    if mode == 'L1':
        return torch.mean(torch.abs(pred-gt), dim=1).cpu().tolist()
    elif mode == 'L2':
        return torch.mean(torch.pow(pred-gt, 2), dim=1).cpu().tolist()
    elif mode == 'cos':
        return torch.cosine_similarity(pred, gt, dim=1).cpu().tolist()
    else:
        raise NotImplementedError






# def visualize_mean_variance(model, dataloader, device=None, epoch=None, mode='train'):
#     model.eval()
#     # cov_list = []
#     # mean_list = []
#     step = 1
#     cov_sum = 0.0
#     mean_sum = 0.0
#     with torch.no_grad():
#         for imgs, _ in dataloader:
#             if device:
#                 imgs = imgs.to(device)
#             pred = model(imgs) # BATCH_SIZE x OUTPUT_DIM
#             output_size = pred.shape[1]
        
#             cov = torch.cov(pred.T) 
            
#             mean = torch.mean(pred, dim=0)

#             # cov_list.append(cov.cpu().detach().numpy())
#             # mean_list.append(mean.cpu().detach().numpy())
#             cov_sum = cov_sum + cov.cpu().detach().numpy()
#             mean_sum = mean_sum + mean.cpu().detach().numpy()
#             step = step + 1

#     # cov_mean = np.around(np.mean(cov_list, axis=0), 3)
#     # mean_mean = np.around(np.mean(mean_list, axis=0), 3)
#     cov_mean = np.around(cov_sum/step, 3)
#     mean_mean = np.around(mean_sum/step, 3)
    
#     fig1 = px.histogram(mean_mean, title='mean'+str(epoch))
#     fig2 = ff.create_annotated_heatmap(cov_mean, x=list(range(output_size)), y=list(range(output_size)), colorscale='Viridis',)

#     fig1.update_layout(showlegend=False)
#     fig2.update_layout(title={'text': 'covariance'+str(epoch)})
#     # if mode == 'train':
#     #     wandb.log({"mean in train": fig1}, commit=False)
#     #     wandb.log({"variance in train": fig2}, commit=False)
#     # else:
#     #     wandb.log({"mean in test": fig1}, commit=False)
#     #     wandb.log({"variance in test": fig2}, commit=False)
#     return fig1, fig2


# def visualize_lenth_angle(data_dict):
#     # data_dict = {'Attack_lenth':[], 'Attack_angle':[], 'Nature_lenth':[], 'Nature_angle':[]}
#     # plot 4 lines in 4 differet colors in one figure; use dash line for angle and solid line for lenth

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=list(range(len(data_dict['Attack_lenth']))), y=data_dict['Attack_lenth'], name='Attack_lenth', line=dict(color='firebrick', width=4)))
#     fig.add_trace(go.Scatter(x=list(range(len(data_dict['Attack_angle']))), y=data_dict['Attack_angle'], name='Attack_angle', line=dict(color='firebrick', width=4, dash='dash')))
#     fig.add_trace(go.Scatter(x=list(range(len(data_dict['Nature_lenth']))), y=data_dict['Nature_lenth'], name='Nature_lenth', line=dict(color='royalblue', width=4)))
#     fig.add_trace(go.Scatter(x=list(range(len(data_dict['Nature_angle']))), y=data_dict['Nature_angle'], name='Nature_angle', line=dict(color='royalblue', width=4, dash='dash')))

#     fig.update_layout(title={'text': 'The p-values of Attack and Nature length and angle'})
#     wandb.log({"The p-values of Attack and Nature length and angle": fig}, commit=False)


  

    
