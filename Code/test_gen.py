import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import pandas as pd
from tqdm import tqdm
import json
import numpy as np


class test_CFG:
    model_name = 'Vanilla_Transformer'
    exp_num = 6
    epoch = 5
    #ckpt_path = f'/Results/{model_name}/{exp_num}/epoch-{epoch}.pt'
    ckpt_path = '/Results/Vanilla_Transformer/9/fold_1/epoch_1.pt'
    #results_path = f'/Results/{model_name}/{exp_num}/epoch_{epoch}'
    results_path = '/Results/Vanilla_Transformer/9/fold_1/epoch_1'
    
    df_path = '/Data/sample_submission.csv'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    valid_path = '/Results/binary_xgb_3_modified_new.csv'
    

all_predictions = {'valid':[],'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}

from get_data import train_dataset, test_dataset
from utils import collate_fn_train, collate_fn_test

test_ds = test_dataset()
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn = collate_fn_test)
test_df = pd.read_csv(test_CFG.df_path)
valid_df = pd.read_csv(test_CFG.valid_path)

# IMPORT MODEL
from Models.Set_Transformer.set_model import set_transformer_model
from Models.Vanilla_Transformer import vanilla_transformer_model
from Config.get_config import get_config


model_CFG = get_config(test_CFG.model_name,test_CFG.exp_num)
if test_CFG.model_name == 'Set_Transformer':
    model = set_transformer_model(**model_CFG,device=test_CFG.device)
    
elif test_CFG.model_name == 'Vanilla_Transformer':
    model = vanilla_transformer_model(**model_CFG,device=test_CFG.device)


    
print(f'Saving at {test_CFG.ckpt_path}') 
   
model.to(device=test_CFG.device)
model.eval()

with torch.no_grad():
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Test Evaluation Loop'):
        batch_x = batch
        del batch_x[0]['kernel_name']
        del batch_x[0]['version']
        # Forward pass through the model
        ypred = model(batch_x)
        ypred_reg = F.softplus(ypred[2:])
        # break
        all_predictions['valid'].append(True if torch.argmax(F.softmax(ypred[:2],dim=0)).item()==1 else 'False')
        all_predictions['latency'].append(ypred_reg[0].item())
        all_predictions['util-BRAM'].append(ypred_reg[1].item())
        all_predictions['util-LUT'].append(ypred_reg[2].item())
        all_predictions['util-FF'].append(ypred_reg[3].item())
        all_predictions['util-DSP'].append(ypred_reg[4].item())
    
with open(test_CFG.results_path+'_predictions.json','w') as f:
    json.dump(all_predictions,f)
    
test_df['valid'] = valid_df['valid']
# test_df['valid'] = all_predictions['valid']
test_df['perf'] = list(map(lambda x:(1e7/np.exp(2*x))-1.0,all_predictions['latency']))
test_df['util-BRAM'] = all_predictions['util-BRAM']
test_df['util-LUT'] = all_predictions['util-LUT']
test_df['util-FF'] = all_predictions['util-FF'] 
test_df['util-DSP'] = all_predictions['util-DSP'] 

test_df.loc[test_df['valid'] == False, 'util-BRAM'] = 0
test_df.loc[test_df['valid'] == False, 'util-LUT'] = 0
test_df.loc[test_df['valid'] == False, 'util-FF'] = 0
test_df.loc[test_df['valid'] == False, 'util-DSP'] = 0
test_df.loc[test_df['valid'] == False, 'perf'] = 0

test_df.to_csv(test_CFG.results_path+'.csv',index=False)
test_df['perf'] = 5
test_df.to_csv(test_CFG.results_path+'_perf0.csv',index=False)
