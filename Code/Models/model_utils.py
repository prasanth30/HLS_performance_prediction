import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from Config.get_config import get_config
import json

from Models.Set_Transformer.set_model import set_transformer_model
from Models.Vanilla_Transformer import vanilla_transformer_model
from Models.transformer_posembed import positional_transformer

from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.nn import functional as F


def save_model(model, ckpt_path, epoch):
    """Save the model at the end of an epoch."""
    model_save_path = os.path.join(ckpt_path, f"epoch_{epoch + 1}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

def save_results(results, ckpt_path):
    """Save the results in JSON format."""
    results_path = os.path.join(ckpt_path, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

def prepare_model(model_name, exp_no, device):
    """Prepare the model, optimizer, and scheduler."""
    model_cfg = get_config(model_name, exp_no)
    model_lr = model_cfg['lr']
    del model_cfg['lr']
    if model_name == 'Set_Transformer':
        model = set_transformer_model(**model_cfg, device=device)
    elif model_name == 'Vanilla_Transformer':
        model = vanilla_transformer_model(**model_cfg, device=device)
    elif model_name == 'Positional_Transformer':
        model = positional_transformer(**model_cfg, device=device)
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=900, T_mult=2)
    return model, optimizer, scheduler

def evaluate_model(model, loader, reg_keys, device):
    """Evaluate the model on validation data."""
    val_targets = {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
    val_preds = {'valid': [], 'latency': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            batch_x = batch['X']
            batch_y = batch['y']
            target_class = batch_y['valid'].to(dtype=torch.long, device=device)
            target_reg = torch.vstack([batch_y[key] for key in reg_keys]).T.to(torch.float32, device=device)

            y_pred = model(batch_x)

            val_preds['valid'].append(torch.argmax(F.softmax(y_pred[:, :2], dim=0)).cpu().numpy())
            for idx, key in enumerate(reg_keys):
                val_preds[key].append(y_pred[:, 2:][idx].cpu().numpy())
            
            for key in reg_keys:
                val_targets[key].extend(batch_y[key].cpu().numpy())
            val_targets['valid'].extend(batch_y['valid'].cpu().numpy())
    
    return val_targets, val_preds


def compute_metrics(val_targets, val_preds, reg_keys):
    f1 = f1_score(val_targets['valid'], val_preds['valid'], average=None)
    rmse = np.mean([np.sqrt(np.mean((np.array(val_preds[key]) - np.array(val_targets[key]))**2)) for key in reg_keys])
    
    val_preds['perf'] = [(1e7 / np.exp(x * 2)) - 1 for x in val_preds['latency']]
    val_targets['perf'] = [(1e7 / np.exp(x * 2)) - 1 for x in val_targets['latency']]
    
    return f1, rmse


def calculate_loss(ypred, target_classification, target_regression, classification_weight, regression_weight, clas_loss_fn, reg_loss_fn):
    classification_loss = clas_loss_fn(ypred[:, :2], target_classification)
    ypred_regression = F.softplus(ypred[:, 2:])
    regression_loss = reg_loss_fn(ypred_regression, target_regression)
    total_loss = classification_weight * classification_loss + regression_weight * regression_loss
    return total_loss, classification_loss, regression_loss

def save_model(model, path, epoch):
    model_save_path = os.path.join(path, f"epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")
    
