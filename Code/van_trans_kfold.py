from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch
import os
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from get_data import train_dataset, test_dataset
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn_train, collate_fn_test, sigmoid, inv_sigmoid
from metric import score

from Models.Set_Transformer.set_model import set_transformer_model
from Models.Vanilla_Transformer import vanilla_transformer_model
from Models.transformer_posembed import positional_transformer
from Config.get_config import get_config

# Assuming you have the other parts (like imports, dataset preparation, etc.) already in place
parser = argparse.ArgumentParser(description="Model Configuration")
parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model')
parser.add_argument('--exp_no', type=int, default=6, help='Experiment number')
parser.add_argument('--model_name', type=str, default='Set_Transformer',choices=['Set_Transformer','Vanilla_Transformer','Positional_Transformer'], help='Name of the model')
parser.add_argument('--reg_wt', type=int, default=1.0, help='Weight of Regression Loss')
parser.add_argument('--class_wt', type=int, default=1.0, help='Weight of Classifier Loss')

args = parser.parse_args()

# TRAINING LOOP
class CFG:
    n_epochs = args.n_epochs
    lr = 1e-5
    exp_no = args.exp_no
    model_name = args.model_name
    ckpt_path = f'/Results/{args.model_name}/{args.exp_no}/'
    device = args.device if torch.cuda.is_available() else 'cpu' 
    regression_weight = args.reg_wt
    classification_weight = args.class_wt
    train_batch_size = 512
    val_batch_size = 1
    
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_CFG = get_config(CFG.model_name,CFG.exp_no)

# Preparing dataset
train_ds = train_dataset(versions=['v20','v21'], split='None', filter='None')
X = [data['X'] for data in train_ds]
y = [data['y']['valid'] for data in train_ds]  # Use the 'valid' field for stratification

ood_f1_scores = []
ood_rmse_scores = []
ood_total_scores = []
reg_keys = ['latency', 'util-BRAM', 'util-LUT', 'util-FF', 'util-DSP']
train_targets =  {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
train_preds = {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'Fold {fold_idx+1}:')

    # Create dataloaders for train and validation for this fold
    train_subset = torch.utils.data.Subset(train_ds, train_idx)
    val_subset = torch.utils.data.Subset(train_ds, val_idx)

    train_dataloader = DataLoader(train_subset, shuffle=True, batch_size=CFG.train_batch_size, collate_fn=collate_fn_train,drop_last=True)
    val_dataloader = DataLoader(val_subset, shuffle=False, batch_size=CFG.val_batch_size, collate_fn=collate_fn_train,drop_last=True)

    # Initialize model, optimizer, and scheduler
    if CFG.model_name == 'Set_Transformer':
        model = set_transformer_model(**model_CFG, device=CFG.device)
    elif CFG.model_name == 'Vanilla_Transformer':
        model = vanilla_transformer_model(**model_CFG, device=CFG.device)
    elif CFG.model_name == 'Positional_Transformer':
        model = positional_transformer(**model_CFG, device= CFG.device)
        

    model.to(device=CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=900, T_mult=2)
    clas_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    fold_ckpt_path = os.path.join(CFG.ckpt_path, f"fold_{fold_idx+1}")
    if not os.path.exists(fold_ckpt_path):
        os.makedirs(fold_ckpt_path)
        
    # Train the model
    model.train()
    for epoch in range(CFG.n_epochs):
        train_loss = 0.0
        i=0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            batch_x = batch['X']
            batch_y = batch['y']
            target_classification = batch_y['valid'].to(dtype=torch.long, device=CFG.device)
            target_regression = torch.vstack([batch_y[reg_key] for reg_key in reg_keys]).T.to(dtype=torch.float32, device=CFG.device)

            ypred = model(batch_x)

            # Calculate losses
            classification_loss = clas_loss_fn(ypred[:, :2], target_classification)
            ypred_regression = F.softplus(ypred[:, 2:])
            regression_loss = reg_loss_fn(ypred_regression, target_regression)
            total_loss = CFG.regression_weight * regression_loss +  CFG.classification_weight*classification_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step(epoch + i/len(train_dataloader))
            train_loss += total_loss.item()
            i+=1
        model_save_path = os.path.join(fold_ckpt_path, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")
    # Evaluate the model on validation fold (OOD evaluation)
        model.eval()
        val_targets = {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
        val_preds = {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                batch_x = batch['X']
                batch_y = batch['y']
                target_classification = batch_y['valid'].to(dtype=torch.long, device=CFG.device)
                target_regression = torch.vstack([batch_y[reg_key] for reg_key in reg_keys]).T.to(dtype=torch.float32, device=CFG.device)

                ypred = model(batch_x)
                # Collect predictions
                valid_pred_class = torch.argmax(F.softmax(ypred[:2],dim=0))
                val_preds['valid'].append(valid_pred_class.cpu().numpy())
                for idx, reg_key in enumerate(reg_keys):
                    val_preds[reg_key].append(ypred[2:][idx].cpu().numpy())

                for key in reg_keys:
                    val_targets[key].extend(batch_y[key].cpu().numpy())
                val_targets['valid'].extend(batch_y['valid'].cpu().numpy())

    # Calculate OOD metrics (F1, RMSE, and custom score) for this fold
        f1 = f1_score(val_targets['valid'], val_preds['valid'], average=None)
        rmse = 0.0
        for key in reg_keys:
            rmse += np.sqrt(np.mean((np.array(val_preds[key]) - np.array(val_targets[key]))**2))
        rmse /= len(reg_keys)
        # Apply custom performance function to latency
        val_preds['perf'] = list(map(lambda x: ((1e7 / np.exp(x * 2)) - 1), val_preds['latency']))
        val_targets['perf'] = list(map(lambda x: ((1e7 / np.exp(x * 2)) - 1), val_targets['latency']))
        
        # Custom scoring function
        _, total_rmse, val_score = score(val_targets, val_preds)
        print(f'Epoch F1:- {f1} RMSE:-{rmse}')
        print(f'From Score function F1:- {_}, RMSE:- {total_rmse},Score:- {val_score}')
    # Append fold results
    ood_f1_scores.append(f1)
    ood_rmse_scores.append(rmse)
    ood_total_scores.append(val_score)

    print(f"Fold {fold_idx+1} Results: F1={_:.4f}, RMSE={rmse:.4f}, Score={val_score:.4f}")

# Calculate average OOD metrics across all folds
mean_f1 = np.mean(ood_f1_scores)
mean_rmse = np.mean(ood_rmse_scores)
mean_score = np.mean(ood_total_scores)

print(f"Average OOD Results across all folds: F1={mean_f1:.4f}, RMSE={mean_rmse:.4f}, Score={mean_score:.4f}")

import json

# Save results to JSON
results = {
    "fold_results": [],
    "mean_f1": mean_f1,
    "mean_rmse": mean_rmse,
    "mean_score": mean_score
}

# Add results from each fold
for i in range(5):
    results["fold_results"].append({
        "fold": i + 1,
        "f1": ood_f1_scores[i].tolist(),  # Convert to list if it's a numpy array
        "rmse": ood_rmse_scores[i],
        "score": ood_total_scores[i]
    })

# Define path to save results
results_path = os.path.join(CFG.ckpt_path, 'results.json')

# Save to JSON file
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {results_path}")
