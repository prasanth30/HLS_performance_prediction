import os
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./../')

from get_data import train_dataset, test_dataset
from utils import collate_fn_train, collate_fn_test

import networkx as nx
from collections import Counter
from tqdm import tqdm

from Set_Transformer.set_model import set_transformer
from Config.set_transformer_config import model_CFG, pragmas

from metric import score
# Assume checkpoint directory exists, otherwise create it
checkpoint_dir = '/Results/checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_ds = train_dataset(versions=['v18','v20'],split='train')
val_ds = train_dataset(versions=['v21'],split='val')
test_ds = test_dataset()

train_dataloader = DataLoader(train_ds, batch_size=1, collate_fn = collate_fn_train)
val_dataloader = DataLoader(val_ds, batch_size=1, collate_fn = collate_fn_train)
test_dataloader = DataLoader(test_ds, batch_size=1, collate_fn = collate_fn_test)   
    
    
model = set_transformer()
model = model.to(device)

clas_loss_fn = nn.CrossEntropyLoss()
reg_loss_fn = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=model_CFG.lr)

train_epochs = []
val_epochs = []

for epoch in range(model_CFG.n_epochs):
    print(f"Training on {epoch}")
    train_losses = []
    train_cl_losses = []
    train_rg_losses = []
    
    # Training Loop
    model.train()
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training Loop'):
        optimizer.zero_grad()
        batch_x = batch['X'][0]
        batch_y = batch['y'][0]
        
        ypred = model(batch_x)
        
        # Classification target and regression targets
        target_classification = torch.tensor(batch_y['valid'], dtype=torch.int64, device=device)
        target_ls = [torch.tensor(v, dtype=torch.float32) for k, v in batch_y.items() if k != 'valid']
        target_regression = torch.tensor(target_ls, device=device)
        
        # Loss calculation
        classification_loss = clas_loss_fn(ypred[:2], target_classification)
        reggression_loss = reg_loss_fn(ypred[2:], target_regression)
        total_loss = model_CFG.regression_weight * reggression_loss + classification_loss
        
        # Backpropagation and optimization
        total_loss.backward()
        optimizer.step()
        
        # Collect losses
        total_loss = total_loss.detach().cpu()
        classification_loss = classification_loss.detach().cpu()
        reggression_loss = reggression_loss.detach().cpu()
        train_losses.append(total_loss)
        train_cl_losses.append(classification_loss)
        train_rg_losses.append(reggression_loss)

    train_epochs.append((train_losses, train_cl_losses, train_rg_losses))
    
    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))
    
    # Validation Loop
    print('Evaluation:-')
    val_losses = []
    val_cl_losses = []
    val_rg_losses = []
    
    model.eval()  # Set model to evaluation mode
    print('Evaluation:-')
    val_losses = []
    val_cl_losses = []
    val_rg_losses = []
    
    all_targets = {'valid': [], 'perf': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
    all_predictions = {'valid': [], 'perf': [], 'util-BRAM': [], 'util-LUT': [], 'util-FF': [], 'util-DSP': []}
    
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Validation Loop'):
            
            batch_x = batch['X'][0]
            batch_y = batch['y'][0]
            
            ypred = model(batch_x)
            
            # Classification target and regression targets
            target_classification = torch.tensor(batch_y['valid'], dtype=torch.int64, device=device)
            target_ls = [torch.tensor(v, dtype=torch.float32) for k, v in batch_y.items() if k != 'valid']
            target_regression = torch.tensor(target_ls, device=device)
            
            # Loss calculation
            classification_loss = clas_loss_fn(ypred[:2], target_classification)
            reggression_loss = reg_loss_fn(ypred[2:], target_regression)
            total_loss = model_CFG.regression_weight * reggression_loss + classification_loss
            
            # Collect losses
            total_loss = total_loss.detach().cpu()
            classification_loss = classification_loss.detach().cpu()
            reggression_loss = reggression_loss.detach().cpu()
            ypred = ypred.detach().cpu()
            
            val_losses.append(total_loss)
            val_cl_losses.append(classification_loss)
            val_rg_losses.append(reggression_loss)
            
            # Accumulate targets and predictions for final score calculation
            all_targets['valid'].append(batch_y['valid'])
            all_targets['perf'].append(batch_y['perf'])
            all_targets['util-BRAM'].append(batch_y['util-BRAM'])
            all_targets['util-LUT'].append(batch_y['util-LUT'])
            all_targets['util-FF'].append(batch_y['util-FF'])
            all_targets['util-DSP'].append(batch_y['util-DSP'])
            
            valid_pred_prob = F.softmax(ypred[:2], dim=0)  # Apply softmax to logits
            valid_pred_class = torch.argmax(valid_pred_prob).item()  # Get predicted class (0 or 1)
            all_predictions['valid'].append(valid_pred_class)
            
            all_predictions['perf'].append(ypred[2])
            all_predictions['util-BRAM'].append(ypred[3])
            all_predictions['util-LUT'].append(ypred[4])
            all_predictions['util-FF'].append(ypred[5])
            all_predictions['util-DSP'].append(ypred[6])

    val_epochs.append((val_losses, val_cl_losses, val_rg_losses))
    
    # Calculate custom score using the score function on the whole validation set
    validation_score = score(all_targets, all_predictions)
    
    # Print average losses and score
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    
    print(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}, Val Score: {validation_score}")
    
    # Plot and save the loss graph
    plt.figure(figsize=(10, 6))
    plt.plot([sum(epoch_losses) / len(epoch_losses) for epoch_losses, _, _ in train_epochs], label='Train Loss')
    plt.plot([sum(epoch_losses) / len(epoch_losses) for epoch_losses, _, _ in val_epochs], label='Val Loss')
    plt.title(f'Loss Over Epochs (Epoch {epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_epoch_{epoch}.png')
    # plt.show()