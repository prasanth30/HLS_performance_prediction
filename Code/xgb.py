import lightgbm as lgb
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import numpy as np
from get_data import train_dataset_kfold
from torch.utils.data import DataLoader
from utils import collate_fn_train, collate_fn_test, pragmas, sigmoid

exp_no = 3

str_to_int = {'off':0.01,'flatten':1,'':0.02,'NA':0.03}

def extract_features(dataset, prag_list):
    X = []
    
    for batch in dataset:
        item=list(batch['X'].values())
        kernel_name = batch['kernel_name']
        
        features = batch['X']
        feature_vector = np.zeros(len(prag_list)+len(pragmas.kernels))
        feature_vector[pragmas.kernels.index(kernel_name)] = 1
        for j, key in enumerate(prag_list):
            i = len(pragmas.kernels) + j
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

models = []  
ood_scores = []
fold = 1

params = {
    'objective': 'binary', 
    # 'metric': '',
    'boosting_type': 'gbdt',
    'learning_rate': 0.3,
    'num_leaves': 400,
    'max_depth': 200,
    'verbose': 1,
}


dataset_kfold = train_dataset_kfold(versions=['v18', 'v20', 'v21'], n_folds=5)
folds = dataset_kfold.get_folds()
for fold_idx, (train_data, valid_data) in enumerate(folds):
    
    train_extr_data = extract_features(train_data,pragmas.pragmas)
    train_extr_target = extract_targets(train_data)
    
    
    valid_extr_data = extract_features(valid_data,pragmas.pragmas)
    valid_extr_target = extract_targets(valid_data)

    train_data_lgb = lgb.Dataset(train_extr_data, label=train_extr_target)
    val_data_lgb = lgb.Dataset(valid_extr_data, label=valid_extr_target, reference=train_data_lgb)

    # Train the model
    gbm = lgb.train(params, train_data_lgb, valid_sets=[val_data_lgb], 
                    num_boost_round=100, callbacks=[lgb.early_stopping(stopping_rounds=10)]
                    )
    
    y_pred_val_fold = gbm.predict(valid_extr_data)
    y_pred_val_fold = np.round(y_pred_val_fold)  # Round predictions to 0 or 1

    # Calculate F1 score for OOD (on dev set)
    f1_val_fold = f1_score(valid_extr_target, y_pred_val_fold)
    ood_scores.append(f1_val_fold)
    
    # Save model for ensembling later
    models.append(gbm)
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