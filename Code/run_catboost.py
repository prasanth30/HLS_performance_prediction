from catboost import CatBoostClassifier, Pool
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import numpy as np
from get_data import train_dataset, test_dataset
from torch.utils.data import DataLoader
from utils import collate_fn_train, collate_fn_test, pragmas, sigmoid

exp_no = 3

str_to_int = {'off':0.01,'flatten':1,'':0.02,'NA':0.03}

def extract_features(dataset, prag_list):
    X = []
    for batch in dataset:
            if 'X' in batch:
                item = batch['X'][0]
            else:
                item = batch[0]
            features = item
            feature_vector = np.zeros(len(prag_list))
            
            for i, key in enumerate(prag_list):
                if key in features:
                    if features[key] in str_to_int:
                        features[key] = str_to_int[features[key]]
                
                    if key in pragmas.integer_space:
                        feature_vector[i] = (int(features[key])-1)/pragmas.mx_range[key]
                    else:
                        feature_vector[i] = features[key]                    
            X.append(feature_vector)
    return np.array(X)


def extract_targets(dataset):
    X = []
    y = []
    for batch in dataset:
        item = batch
        target = item['y']
        y.append(1 if target['valid'] else 0)
    return np.array(y)


train_ds = train_dataset(versions=['v20','v21'],split='None')
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,collate_fn=collate_fn_train)
X = extract_features(train_loader, pragmas.pragmas)
y = extract_targets(train_loader)


test_ds = test_dataset()
test_dataloader = DataLoader(test_ds,batch_size=1,shuffle=False,collate_fn=collate_fn_test)
X_test = extract_features(test_dataloader,pragmas.pragmas)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = []  
ood_scores = []
fold = 1

params = {
    'iterations': 20, 
    'learning_rate': 0.03,
    'depth': 16, 
    'loss_function': 'Logloss',
    'eval_metric': 'F1',
    'verbose': True,
}

for train_index, val_index in kf.split(X, y):
    X_train, X_val_fold = X[train_index], X[val_index]
    y_train, y_val_fold = y[train_index], y[val_index]

    train_data_cb = Pool(X_train, label=y_train)
    val_data_cb = Pool(X_val_fold, label=y_val_fold)

    cbm = CatBoostClassifier(**params)
    # Train the model
    cbm.fit(train_data_cb, eval_set=val_data_cb, early_stopping_rounds=10)
    
    y_pred_val_fold = cbm.predict(X_val_fold)
    y_pred_val_fold = np.round(y_pred_val_fold)  # Round predictions to 0 or 1

    # Calculate F1 score for OOD (on dev set)
    f1_val_fold = f1_score(y_val_fold, y_pred_val_fold)
    ood_scores.append(f1_val_fold)
    
    # Save model for ensembling later
    models.append(cbm)
    print(f'Fold {fold}: F1 Score on dev set: {f1_val_fold:.4f}')
    fold += 1

ood_avg_score = np.mean(ood_scores)
print(f'Average OOD F1 Score across all folds: {ood_avg_score:.4f}')

# Train a LightGBM model
# train_data_lgb = lgb.Dataset(X_train, label=y_train)
# dev_data_lgb = lgb.Dataset(X_dev,label=y_dev)
# valid_data_lgb = lgb.Dataset(X_val, label=y_val)

# print(len(train_data_lgb.data))
# print((train_data_lgb.data[0].shape))
y_pred_test_ensemble = np.zeros(len(X_test))

for model in models:
    pr = model.predict(X_test) 
    y_pred_test_ensemble += pr
    print(pr)
# Average predictions and apply threshold for binary classification
y_pred_test_ensemble /= len(models)
y_pred_test_ensemble = np.where(y_pred_test_ensemble > 0.5, True, False)

import pandas as pd
df = pd.read_csv('/Data/sample_submission.csv')

df['perf'] = 5
df['valid'] = y_pred_test_ensemble

df.to_csv(f'/Results/binary_xgb_{exp_no}.csv',index=False)

df.loc[df['valid']==False,'perf'] = 0

df.to_csv(f'/Results/binary_xgb_{exp_no}_modified.csv',index=False)
