import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from scipy.stats import gaussian_kde, norm
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import plotly.figure_factory as ff  
import plotly.express as px
import plotly.graph_objects as go

from src.model import ResidualBlock,ResNet
from src.hypotest import multi_checkmean, multi_checkcov, ks_test

from src.dataset import CusDataset
from src.utils import set_random_seed, load_model
from src.test_module import unsuccess_ratio

set_random_seed(302)


def test_normality (dataloader, model, device=None):
    num_batches = len(dataloader)
    model.eval()
    
    mean_p_list = []
    cov_p_list = []
    pred_list = []
    norm_p_list = []

    with torch.no_grad():
        for imgs, _ in dataloader:
            
            if device:
                imgs = imgs.to(device)
            pred = model(imgs)
            pred_list.append(pred)

            tested_obs = pred.clone().detach().to('cpu')
        
            # for the mean hypothesis test
            mean_p = multi_checkmean(tested_obs, mean=0.0, confidence = 0.05)
            mean_p_list.append(mean_p)

            # for the covariance hypothesis test
            cov_p = multi_checkcov(tested_obs, cov=None, confidence = 0.05)
            cov_p_list.append(cov_p)

    preds = torch.stack(pred_list).clone().detach().to('cpu')
    norm_p_list = ks_test(preds.reshape(-1, preds.size(-1)).t())
    pvalues = {'mean_pvalues': mean_p_list, 'cov_pvalues': cov_p_list, 'norm_pvalues': norm_p_list}

    return pvalues, preds


device = "cuda" if torch.cuda.is_available() else "cpu"

transform_test = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])
test_batch_size = 100
test_img_folder =  '../data/coco2017/images/val2017'
test_data = CusDataset(root_dir=test_img_folder, train=False, transform=transform_test, if_save=True)
transnorm = {'mean':[0.5, 0.5, 0.5], 'std':[0.5, 0.5, 0.5]} 
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=4)

output_size = 32
load_name = "./stored_models/examplemodel_coco_resnet18.pt"
wmmodel = ResNet(ResidualBlock, output_size=output_size, transnorm=transnorm)
wmmodel = load_model(wmmodel, load_name, prefix='_orig_mod.')
wmmodel = wmmodel.to(device)

results = test_normality(test_dataloader, wmmodel, device=device)

pvalues = unsuccess_ratio(results[0], confidence=0.05, ifprint=False)

print('Percentage of mean vector being zero vector: {}%'.format((1-pvalues['mean_pvalues'][0])*100)) # p<0.05 means the output is not zero vector
print('Percentage of covariance matrix being identity matrix: {}%'.format((1-pvalues['cov_pvalues'][0])*100)) # p<0.05 means the output is not identity matrix
print('Percentage of entries following normality in ouput vector {}%'.format((1-pvalues['norm_pvalues'][0])*100)) # p<0.05 means the output is not normal distribution


###########ploting the mean and covariance of the output###########
output_preds = results[1]
mean_vector = torch.mean(output_preds, dim=1)
cov_matrix = []
for i in range(output_preds.shape[0]):
    cov= torch.cov(output_preds[i].T)
    cov_matrix.append(cov)
    
cov_matrix = torch.stack(cov_matrix)

cov_mean = cov_matrix.mean(dim=0).cpu().detach().numpy()
mean_mean = mean_vector.mean(dim=0).cpu().detach().numpy()

cov_mean = np.round(cov_mean, decimals=2)
mean_mean = np.round(mean_mean, decimals=2)

order = np.argsort(np.diag(cov_mean))
cov_mean = cov_mean[order][:, order]

output_size = mean_mean.shape[0]
fig1 = px.histogram(mean_mean, title=None, color_discrete_sequence=['#2f466e'])
fig2 = ff.create_annotated_heatmap(cov_mean, x=list(range(output_size)), y=list(range(output_size))[::-1],colorscale='cividis')

layout_config_mean=go.Layout(paper_bgcolor='rgba(255,255,255)',
                                plot_bgcolor='rgba(255,255,255, 0)',
                                xaxis=dict(linecolor='black', linewidth=1, mirror=False),
                                yaxis=dict(linecolor='black', linewidth=1,mirror=False,
                                            gridwidth=1, gridcolor='rgba(0,0,0,0.1)', griddash='dot',
                                            showgrid=True))
layout_config_cov=go.Layout(paper_bgcolor='rgba(255,255,255)', 
                            plot_bgcolor='rgba(255,255,255, 0)',
                            xaxis=dict(showticklabels=False, zeroline=False,),  
                            yaxis=dict( showticklabels=False, zeroline=False,))  
for annotation in fig2.layout.annotations:
    annotation.font.size = 5
    
fig1.update_layout(layout_config_mean, title_x=0.5, title_y=0.85 )
fig2.update_layout(layout_config_cov, title_x=0.5, title_y=0.85)

fig1.update_layout(showlegend=False)

font_size_title = 20
font_size_ticks = 20

axis_layout_config = {
    'xaxis_title_font': {'size': font_size_title},
    'yaxis_title_font': {'size': font_size_title},
    'xaxis_tickfont': {'size': font_size_ticks},
    'yaxis_tickfont': {'size': font_size_ticks},
}

fig1.update_layout(axis_layout_config)

fig1.update_layout(margin=dict(l=20, r=10, t=0, b=5))
fig2.update_layout(margin=dict(l=1, r=1, t=1, b=1))

os.makedirs('./outputs/normality/', exist_ok=True)
fig1.write_image('./outputs/normality/test_mean.pdf',scale=10)
fig2.write_image('./outputs/normality/test_cov.pdf',scale=10)



######### ploting the distribution of the output #########
xaxis_dict = dict(
        linecolor='rgba(0, 0, 0, 1)',
        linewidth=2,
        tickcolor='rgba(0, 0, 0, 1)',
        tickfont=dict(color='rgba(0, 0, 0, 1)', size=30),
        titlefont=dict(color='rgba(0, 0, 0, 1)', size=35),
        mirror=False,
        title_standoff=10,
        )

yaxis=dict(
        linecolor='rgba(0, 0, 0, 1)',
        linewidth=2,
        tickcolor='rgba(0, 0, 0, 1)',
        tickfont=dict(color='rgba(0, 0, 0, 1)',size=30),
        titlefont=dict(color='rgba(0, 0, 0, 1)',size=35),
        mirror=False,
        title_standoff=20,
        )

legend_dict=dict(orientation="h", 
        yanchor="bottom", y=1.02, xanchor="right", x=1, 
        font=dict(size=30, color='rgba(0, 0, 0, 1)'),
        )

layout_config = go.Layout(
paper_bgcolor='rgba(255,255,255)',
plot_bgcolor='rgba(255,255,255, 0)',
xaxis=xaxis_dict,
yaxis=yaxis,
legend=legend_dict,
margin=dict(l=0, r=1, t=0, b=0),
)

    
    
output_preds = output_preds.detach().cpu().numpy().reshape(-1, output_preds.shape[-1])

x = np.linspace(-4, 4, 1000)
y_standard_normal = norm.pdf(x, 0, 1)

kdes = [gaussian_kde(vec) for vec in output_preds.T]
y_vals = np.array([kde(x) for kde in kdes])
y_mean, y_max, y_min = y_vals.mean(axis=0), y_vals.max(axis=0), y_vals.min(axis=0)

fig = go.Figure([
    go.Scatter(x=x, y=y_standard_normal, mode='lines', name=r'$\mathcal{N}(0,1)$', line=dict(color='black', dash='dot')),
    go.Scatter(x=x, y=y_mean, mode='lines', name='Mean', line=dict(color='blue')),
    go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([y_max, y_min[::-1]]), fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='Range')
])



# set axixes title
fig.update_layout(
    xaxis_title=r'$\mathbf{x}_j$',
    yaxis_title=r'$p(\mathbf{x}_j)$',
)
fig.update_layout(layout_config)

legend_dict=dict(x=1, y=1, xanchor='auto', yanchor='auto', orientation="v")
fig.update_layout(legend=legend_dict)
fig.update_layout(            
    xaxis=dict(
        titlefont=dict(size=100), 
    ),
    yaxis=dict(
        titlefont=dict(size=30),  
    )
)

fig.write_image('./outputs/normality/test_normality.pdf', scale=10)








