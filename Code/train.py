from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
# IMPORT DATA
from get_data import train_dataset, test_dataset
from torch.utils.data import DataLoader
from utils import collate_fn_train, collate_fn_test
from metric import score


parser = argparse.ArgumentParser(description="Model Configuration")
parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model')
parser.add_argument('--exp_no', type=int, default=6, help='Experiment number')
parser.add_argument('--model_name', type=str, default='Set_Transformer',choices=['Set_Transformer','Vanilla_Transformer'], help='Name of the model')
parser.add_argument('--reg_wt', type=int, default=1.0, help='Weight of Regression Classifier')
args = parser.parse_args()

# TRAINING LOOP
class CFG:
    n_epochs = args.n_epochs
    lr = 5e-5
    exp_no = args.exp_no
    model_name = args.model_name
    ckpt_path = f'/Results/{args.model_name}/{args.exp_no}/'
    device = args.device if torch.cuda.is_available() else 'cpu' 
    regression_weight = args.reg_wt
    train_batch_size = 32
    val_batch_size = 1


# Define stratified K-fold split
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

logs = {
    'fold_wise_train_loss': [],
    'fold_wise_val_loss': [],
    'fold_wise_ood_scores': [],
}

# train_ds = train_dataset(versions=['v18','v20','v21'],split='train',filter='None')
# valid_ds = train_dataset(versions=['v21'],split='valid',filter='None')
# print(f'Length of Train {len(train_ds)} and Valid {len(valid_ds)}')
# train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=CFG.train_batch_size, collate_fn=collate_fn_train)
# valid_dataloader = DataLoader(valid_ds, shuffle=False, batch_size=CFG.val_batch_size, collate_fn=collate_fn_train)

# # IMPORT MODEL
# from Models.Set_Transformer.set_model import set_transformer_model
# from Models.Vanilla_Transformer import vanilla_transformer_model
# from Config.get_config import get_config


# model_CFG = get_config(CFG.model_name,CFG.exp_no)
# if CFG.model_name == 'Set_Transformer':
#     model = set_transformer_model(**model_CFG,device=CFG.device)
    
# elif CFG.model_name == 'Vanilla_Transformer':
#     model = vanilla_transformer_model(**model_CFG,device=CFG.device)

    
# print(f'Saving at {CFG.ckpt_path}') 
   
# model.to(device=CFG.device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=900, T_mult=2)
# clas_loss_fn = nn.CrossEntropyLoss()
# reg_loss_fn = nn.MSELoss()

# logs = {'batch_wise_train_loss':[],
#         'batch_wise_val_loss':[],
#         'epoch_wise_train_scores':[],
#         'epoch_wise_val_scores':[]
#     }

# for epoch in range(CFG.n_epochs):
#     print('Training Loop:- ')
#     model.train()
    
#     train_losses = []
#     train_cl_losses = []
#     train_rg_losses = []
#     reg_keys = ['latency', 'util-BRAM', 'util-LUT', 'util-FF', 'util-DSP']
#     train_targets =  {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
#     train_preds = {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
#     for idx,batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader),desc ="Training Loop"):
        
#         optimizer.zero_grad()
#         batch_x = batch['X']
#         batch_y = batch['y']
#         target_classification = batch_y['valid'].to(dtype=torch.long,device=CFG.device)
#         target_ls = [batch_y[tar] for tar in reg_keys]
#         target_regression = torch.vstack(target_ls).T.to(dtype=torch.float32,device=CFG.device)
        
#         ypred = model(batch_x)

#         classification_loss = clas_loss_fn(ypred[:,:2], target_classification)

#         ypred_regression = F.softplus(ypred[:,2:])
#         reggression_loss = reg_loss_fn(ypred_regression, target_regression)

#         total_loss = CFG.regression_weight * reggression_loss + classification_loss

#         total_loss.backward()
#         optimizer.step()
#         scheduler.step(epoch + idx / len(train_dataloader))

#         total_loss = total_loss.detach().cpu()
#         classification_loss = classification_loss.detach().cpu()
#         regression_loss = reggression_loss.detach().cpu()
#         ypred_regression = ypred_regression.detach().cpu()
#         ypred = ypred.detach().cpu()

#         train_losses.append(total_loss)
#         train_cl_losses.append(classification_loss)
#         train_rg_losses.append(reggression_loss)

#         train_targets['valid'].extend(batch_y['valid'])
#         for key in reg_keys:
#             train_targets[key].extend(batch_y[key])

#         valid_pred_prob = F.softmax(ypred[:,:2], dim=1)  # Apply softmax to logits
#         valid_pred_class = torch.argmax(valid_pred_prob,dim=1) # Get predicted class (0 or 1)
#         train_preds['valid'].extend(valid_pred_class)      
          
#         for idx,key in enumerate(reg_keys):
#             train_preds[key].extend(ypred_regression[:,idx])

#     logs['batch_wise_train_loss'].append({'train_losses': train_losses, 
#                                          'train_cl_losses': train_cl_losses, 
#                                          'train_rg_losses': train_rg_losses
#                                          })
#     print('Saving Model:- ')
#     if not os.path.exists(CFG.ckpt_path):
#         os.makedirs(CFG.ckpt_path)
        
#     torch.save(model.state_dict(), os.path.join(CFG.ckpt_path, 'epoch-{}.pt'.format(epoch)))
#     print('Validation Loop:- ')
#     val_losses = []
#     val_cl_losses = []
#     val_rg_losses = []
    
#     all_targets = {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
#     all_predictions = {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
    
#     model.eval()
#     with torch.no_grad():  # Disable gradient calculation for validation
#         for idx,batch in tqdm(enumerate(valid_dataloader),total=len(valid_dataloader),desc ="Valid Loop"):
               
#             batch_x = batch['X']
#             batch_y = batch['y']
#             target_classification = batch_y['valid'].to(dtype=torch.long,device=CFG.device)
#             target_ls = [batch_y[tar] for tar in reg_keys]
#             target_regression = torch.vstack(target_ls).T.to(dtype=torch.float32,device=CFG.device)
            
#             ypred = model(batch_x).reshape(-1,7)
            
#             classification_loss = clas_loss_fn(ypred[:,:2], target_classification)
            
#             ypred_regression = F.softplus(ypred[:,2:])
#             reggression_loss = reg_loss_fn(ypred_regression, target_regression)
            
#             total_loss = CFG.regression_weight * reggression_loss + classification_loss
            
#             # Collect losses
#             total_loss = total_loss.detach().cpu()
#             classification_loss = classification_loss.detach().cpu()
#             reggression_loss = reggression_loss.detach().cpu()
#             ypred = ypred.detach().cpu()
#             ypred_regression = ypred_regression.detach().cpu()
            
#             val_losses.append(total_loss)
#             val_cl_losses.append(classification_loss)
#             val_rg_losses.append(reggression_loss)

#             all_targets['valid'].extend(batch_y['valid'])            
#             for key in reg_keys:
#                 all_targets[key].extend(batch_y[key])
            
#             valid_pred_prob = F.softmax(ypred[:,:2], dim=1)  # Apply softmax to logits
#             valid_pred_class = torch.argmax(valid_pred_prob,dim=1)  # Get predicted class (0 or 1)
#             all_predictions['valid'].extend(valid_pred_class)
            
#             for idx,key in enumerate(reg_keys):
#                 all_predictions[key].extend(ypred_regression[:,idx])


#     logs['batch_wise_val_loss'].append({'valid_losses': val_losses, 
#                                          'valid_cl_losses': val_cl_losses, 
#                                          'valid_rg_losses': val_rg_losses
#                                          })
    
#     # Calculate custom score using the score function on the whole validation set
#     #f1_score, validation_score = score(all_targets, all_predictions)
    
#     # Print average losses and score
#     rs = 0.0
#     for key,ls in train_preds.items():
#         if key not in ['perf','valid']:
#             np_pred = np.array(ls)
#             np_tar = np.array(train_targets[key])
#             rs+= np.sqrt(np.mean((np_pred-np_tar)**2))
#     avg_train_loss = rs #sum(train_losses) / len(train_losses)
    
#     rs = 0.0
#     for key,ls in all_predictions.items():
#         if key not in ['perf','valid']:
#             np_pred = np.array(ls)
#             np_tar = np.array(all_targets[key])
#             rs+= np.sqrt(np.mean((np_pred-np_tar)**2))
#     avg_val_loss = rs #sum(val_losses) / len(val_losses)
    
#     all_predictions['perf'] = list(map(lambda x:((1e7/np.exp(x*2))-1),all_predictions['latency']))    
#     all_targets['perf'] = list(map(lambda x:((1e7/np.exp(x*2))-1),all_targets['latency']))
    
#     all_targets['valid'] = list(map(bool,all_targets['valid']))
#     del all_predictions['latency']
#     del all_targets['latency']
#     f1_score, total_rmse, val_score = score(all_targets,all_predictions)
    
#     print(f"Epoch {epoch}:")
#     print(f"Avg Train Loss: {avg_train_loss}, Avg Val Loss:  {avg_val_loss}")
#     print(f"Val F1 Score:   {f1_score}, Val RMSE Score:- {total_rmse}, Val score:- {val_score}")
    
    
    

# Log for storing performance of each fold

# Arrays to store final predictions and targets for OOD evaluation
final_ood_targets = {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
final_ood_predictions = {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}

# Get feature and target arrays
train_ds = train_dataset(versions=['v18', 'v20', 'v21'], split='train', filter='None')
X, y = extract_features_targets(train_ds)  # Function to extract features and labels for stratification

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y['valid'])):
    print(f"Starting Fold {fold + 1}")
    
    # Create dataloaders for the current fold
    train_fold_ds = Subset(train_ds, train_idx)
    val_fold_ds = Subset(train_ds, val_idx)
    
    train_dataloader = DataLoader(train_fold_ds, shuffle=True, batch_size=CFG.train_batch_size, collate_fn=collate_fn_train)
    valid_dataloader = DataLoader(val_fold_ds, shuffle=False, batch_size=CFG.val_batch_size, collate_fn=collate_fn_train)

    # Initialize model, optimizer, and scheduler for each fold
    model = model(**model_CFG, device=CFG.device) if CFG.model_name == 'Set_Transformer' else vanilla_transformer_model(**model_CFG, device=CFG.device)
    model.to(device=CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=900, T_mult=2)
    
    # Training loop for this fold
    for epoch in range(CFG.n_epochs):
        print(f"Training Epoch {epoch + 1} for Fold {fold + 1}")
        model.train()
        train_losses, train_preds, train_targets = [], [], []
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_x, batch_y = batch['X'], batch['y']
            target_classification = batch_y['valid'].to(dtype=torch.long, device=CFG.device)
            target_regression = torch.vstack([batch_y[key] for key in reg_keys]).T.to(dtype=torch.float32, device=CFG.device)
            
            # Forward pass
            y_pred = model(batch_x)
            classification_loss = clas_loss_fn(y_pred[:, :2], target_classification)
            regression_loss = reg_loss_fn(F.softplus(y_pred[:, 2:]), target_regression)
            total_loss = CFG.regression_weight * regression_loss + classification_loss
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            scheduler.step(epoch + len(train_dataloader))

            # Store train losses and targets/predictions for analysis
            train_losses.append(total_loss.item())
            train_preds.append(y_pred.detach().cpu())
            train_targets.append(target_regression.detach().cpu())
        
        # Validation loop
        model.eval()
        val_losses, val_preds, val_targets = [], [], []
        with torch.no_grad():
            for batch in valid_dataloader:
                batch_x, batch_y = batch['X'], batch['y']
                target_classification = batch_y['valid'].to(dtype=torch.long, device=CFG.device)
                target_regression = torch.vstack([batch_y[key] for key in reg_keys]).T.to(dtype=torch.float32, device=CFG.device)
                
                # Forward pass (no gradients)
                y_pred = model(batch_x)
                classification_loss = clas_loss_fn(y_pred[:, :2], target_classification)
                regression_loss = reg_loss_fn(F.softplus(y_pred[:, 2:]), target_regression)
                total_loss = CFG.regression_weight * regression_loss + classification_loss

                # Store val losses and predictions for analysis
                val_losses.append(total_loss.item())
                val_preds.append(y_pred.cpu())
                val_targets.append(target_regression.cpu())

        # Log training and validation losses for the current fold
        logs['fold_wise_train_loss'].append(np.mean(train_losses))
        logs['fold_wise_val_loss'].append(np.mean(val_losses))

        # Perform OOD evaluation (on dev set or external OOD set)
        ood_eval_set = test_dataset()  # Assuming you have an OOD dataset
        ood_dataloader = DataLoader(ood_eval_set, batch_size=CFG.val_batch_size, shuffle=False, collate_fn=collate_fn_test)
        ood_scores = []

        with torch.no_grad():
            for batch in ood_dataloader:
                batch_x = batch['X']
                y_pred = model(batch_x)
                ood_scores.append(score(batch, y_pred.cpu()))  # Custom OOD evaluation function
        
        logs['fold_wise_ood_scores'].append(np.mean(ood_scores))

    print(f"Fold {fold + 1} completed.")

# Aggregate and log final results across all folds
avg_train_loss = np.mean(logs['fold_wise_train_loss'])
avg_val_loss = np.mean(logs['fold_wise_val_loss'])
avg_ood_score = np.mean(logs['fold_wise_ood_scores'])

print(f"Average Training Loss: {avg_train_loss}")
print(f"Average Validation Loss: {avg_val_loss}")
print(f"Average OOD Score: {avg_ood_score}")
