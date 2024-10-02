import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import tqdm
import json
import numpy as np

class test_CFG:
    model_name = 'Positional_Transformer'
    exp_num = 2
    epoch = 1
    num_folds = 5
    ckpt_paths = [f'/Results/Positional_Transformer/{2}/fold_{i+1}/epoch_{1}.pt' for i in range(num_folds)]
    results_path = f'/Results/{model_name}/{exp_num}/{epoch}_final_results'
    
    df_path = '/Data/sample_submission.csv'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    valid_path = '/Results/binary_xgb_3_modified_new.csv'

from get_data import train_dataset, test_dataset
from utils import collate_fn_train, collate_fn_test

test_ds = test_dataset()
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn_test)
test_df = pd.read_csv(test_CFG.df_path)
valid_df = pd.read_csv(test_CFG.valid_path)

all_predictions = {
    'valid': np.zeros((len(test_dataset()), test_CFG.num_folds)),
    'latency': np.zeros((len(test_dataset()), test_CFG.num_folds)),
    'util-BRAM': np.zeros((len(test_dataset()), test_CFG.num_folds)),
    'util-LUT': np.zeros((len(test_dataset()), test_CFG.num_folds)),
    'util-FF': np.zeros((len(test_dataset()), test_CFG.num_folds)),
    'util-DSP': np.zeros((len(test_dataset()), test_CFG.num_folds))
}


# IMPORT MODEL
from Models.Set_Transformer.set_model import set_transformer_model
from Models.Vanilla_Transformer import vanilla_transformer_model
from Models.transformer_posembed import positional_transformer
from Config.get_config import get_config

model_CFG = get_config(test_CFG.model_name, test_CFG.exp_num)

# Loop through each fold
for fold_idx, ckpt_path in enumerate(test_CFG.ckpt_paths):
    if test_CFG.model_name == 'Set_Transformer':
        model = set_transformer_model(**model_CFG, device=test_CFG.device)
    elif test_CFG.model_name == 'Vanilla_Transformer':
        model = vanilla_transformer_model(**model_CFG, device=test_CFG.device)
    elif test_CFG.model_name == 'Positional_Transformer':
        model = positional_transformer(**model_CFG, device= test_CFG.device)
        
    # Load the checkpoint for the current fold
    print(f'Loading checkpoint from {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device=test_CFG.device)
    model.eval()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f'Test Evaluation Loop Fold {fold_idx + 1}'):
            batch_x = batch['X']
            del batch_x[0]['kernel_name']
            del batch_x[0]['version']

            # Forward pass through the model
            ypred = model(batch_x)
            ypred_reg = F.softplus(ypred[2:])

            # Collect predictions for fold
            all_predictions['valid'][i, fold_idx] = 1 if torch.argmax(F.softmax(ypred[:2], dim=0)).item() == 1 else 0
            all_predictions['latency'][i, fold_idx] = ypred_reg[0].item()
            all_predictions['util-BRAM'][i, fold_idx] = ypred_reg[1].item()
            all_predictions['util-LUT'][i, fold_idx] = ypred_reg[2].item()
            all_predictions['util-FF'][i, fold_idx] = ypred_reg[3].item()
            all_predictions['util-DSP'][i, fold_idx] = ypred_reg[4].item()

# Aggregate results: average or max voting
final_predictions = {
    'valid': (np.mean(all_predictions['valid'], axis=1) > 0.5).astype(int),  # Majority vote
    'latency': np.mean(all_predictions['latency'], axis=1),                  # Average
    'util-BRAM': np.mean(all_predictions['util-BRAM'], axis=1),              # Average
    'util-LUT': np.mean(all_predictions['util-LUT'], axis=1),                # Average
    'util-FF': np.mean(all_predictions['util-FF'], axis=1),                  # Average
    'util-DSP': np.mean(all_predictions['util-DSP'], axis=1)                 # Average
}

# Save final predictions to JSON
# with open(test_CFG.results_path + '_final_predictions.json', 'w') as f:
#     json.dump(final_predictions, f)

# Post-processing and saving the final CSV
# test_df['valid'] = final_predictions['valid']
test_df['valid'] = valid_df['valid']
test_df['perf'] = list(map(lambda x: (1e7 / np.exp(2 * x)) - 1.0, final_predictions['latency']))
test_df['util-BRAM'] = final_predictions['util-BRAM']
test_df['util-LUT'] = final_predictions['util-LUT']
test_df['util-FF'] = final_predictions['util-FF']
test_df['util-DSP'] = final_predictions['util-DSP']

# Set utility values to 0 where valid == False
test_df.loc[test_df['valid'] == 0, ['util-BRAM', 'util-LUT', 'util-FF', 'util-DSP', 'perf']] = 0

# Save final results to CSV
test_df.to_csv(test_CFG.results_path + '_final.csv', index=False)

# Optionally set 'perf' to 5 and save a separate file
test_df['perf'] = 5
test_df.to_csv(test_CFG.results_path + '_perf5.csv', index=False)
