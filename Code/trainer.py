import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import collate_fn_train
from Models.model_utils import save_model, save_results, prepare_model, evaluate_model, compute_metrics
from metric import score
from get_data import train_dataset_kfold

parser = argparse.ArgumentParser(description="Model Configuration")
parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model')
parser.add_argument('--exp_no', type=int, default=6, help='Experiment number')
parser.add_argument('--model_name', type=str, default='Set_Transformer',choices=['Set_Transformer','Vanilla_Transformer','Positional_Transformer'], help='Name of the model')
parser.add_argument('--reg_wt', type=float, default=1.0, help='Weight of Regression Loss')
parser.add_argument('--class_wt', type=float, default=1.0, help='Weight of Classification Loss')

args = parser.parse_args()

class CFG:
    n_epochs = args.n_epochs
    exp_no = args.exp_no
    model_name = args.model_name
    ckpt_path = f'/Results/{args.model_name}/{args.exp_no}/'
    device = args.device if torch.cuda.is_available() else 'cpu' 
    regression_weight = args.reg_wt
    classification_weight = args.class_wt
    train_batch_size = 512
    val_batch_size = 1

# Define StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
reg_keys = ['latency', 'util-BRAM', 'util-LUT', 'util-FF', 'util-DSP']

ood_f1_scores = []
ood_rmse_scores = []
ood_total_scores = []

# Get dataset and fold splits
dataset_kfold = train_dataset_kfold(versions=['v18', 'v20', 'v21'], n_folds=5)
folds = dataset_kfold.get_folds()

for fold_idx, (train_data, valid_data) in enumerate(folds):
    print(f"Fold {fold_idx + 1}:")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=CFG.train_batch_size, collate_fn=collate_fn_train, drop_last=True, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=CFG.val_batch_size, collate_fn=collate_fn_train, drop_last=False, shuffle=False)

    # Prepare the model
    model, optimizer, scheduler = prepare_model(CFG.model_name, args.exp_no, CFG.device)

    fold_ckpt_path = os.path.join(CFG.ckpt_path, f"fold_{fold_idx+1}")
    os.makedirs(fold_ckpt_path, exist_ok=True)

    # Training Loop
    model.train()
    for epoch in range(CFG.n_epochs):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()

            batch_x, batch_y = batch['X'], batch['y']
            target_class = batch_y['valid'].to(dtype=torch.long, device=CFG.device)
            target_reg = torch.vstack([batch_y[key] for key in reg_keys]).T.to(dtype=torch.float32, device=CFG.device)

            # Forward pass
            y_pred = model(batch_x)
            class_loss = nn.CrossEntropyLoss()(y_pred[:, :2], target_class)
            reg_loss = nn.MSELoss()(F.softplus(y_pred[:, 2:]), target_reg)

            # Total loss
            total_loss = CFG.regression_weight * reg_loss + CFG.classification_weight * class_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()

        # Save model after each epoch
        save_model(model, fold_ckpt_path, epoch)

        # Validation after each epoch
        model.eval()
        val_targets, val_preds = evaluate_model(model, valid_loader, reg_keys, CFG.device)
        f1, rmse, val_score = compute_metrics(val_targets, val_preds, reg_keys)

        print(f'Epoch {epoch + 1} - F1: {f1:.4f}, RMSE: {rmse:.4f}, Score: {val_score:.4f}')

    # Store fold results
    ood_f1_scores.append(f1)
    ood_rmse_scores.append(rmse)
    ood_total_scores.append(val_score)

# Calculate averages across all folds
mean_f1 = np.mean(ood_f1_scores)
mean_rmse = np.mean(ood_rmse_scores)
mean_score = np.mean(ood_total_scores)

# Save results
results = {
    "mean_f1": mean_f1,
    "mean_rmse": mean_rmse,
    "mean_score": mean_score,
    "fold_results": [{"fold": i + 1, "f1": f, "rmse": r, "score": s} for i, (f, r, s) in enumerate(zip(ood_f1_scores, ood_rmse_scores, ood_total_scores))]
}
save_results(results, CFG.ckpt_path)
