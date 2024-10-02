import numpy as np

def score(targets, predictions):
    """
    Evaluates model performance using ground truth (targets) and predicted values (predictions).
    Both inputs are expected to be dictionaries with the same structure.
    
    Arguments:
    - targets: Dictionary with keys ('valid', 'perf', 'util-BRAM', 'util-LUT', 'util-FF', 'util-DSP') and their respective true values.
    - predictions: Dictionary with the same keys and predicted values.
    
    Returns:
    - score: Float score based on classification (F1 score) and regression (RMSE) evaluation.
    """
    # Initialize classification metrics for 'valid' (True/False classification)
    tp, tn, fp, fn = 0, 0, 0, 0
    for pred, truth in zip(predictions['valid'], targets['valid']):
        if pred == True and truth == True:
            tp += 1
        elif pred == True and truth == False:
            fp += 1
        elif pred == False and truth == True:
            fn += 1
        else:
            tn += 1

    # Compute F1 score for 'valid'
    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-6)  # Add small epsilon to avoid division by zero

    # Regression (RMSE) for 'perf' considering only valid instances (valid == True in both targets and predictions)
    valid_mask = np.array(targets['valid']) # Mask for valid entries in targets

    perf_target = np.array(targets['perf'])#[valid_mask]  # Filter only valid entries
    perf_pred = np.array(predictions['perf'])#[valid_mask]  # Filter only valid entries
    
    perf_target[valid_mask] = 0
    
    if len(perf_target) > 0:
        rmse_perf = np.sqrt(np.mean((np.log(1e7 / (perf_pred + 1)) / 2 - np.log(1e7 / (perf_target + 1)) / 2) ** 2))
    else:
        rmse_perf = 0  # Handle cases where no valid entries exist

    # RMSE for 'util-BRAM', 'util-LUT', 'util-FF', 'util-DSP' considering only valid instances
    rmse_util = 0
    for util_key in ['util-BRAM', 'util-LUT', 'util-FF', 'util-DSP']:
        util_target = np.array(targets[util_key])[valid_mask]  # Filter only valid entries
        util_pred = np.array(predictions[util_key])[valid_mask]  # Filter only valid entries
        
        if len(util_target) > 0:
            rmse_util += np.sqrt(np.mean((util_pred - util_target) ** 2))

    # Combine the RMSE for 'perf' and 'util'
    total_rmse = rmse_perf + rmse_util

    # Final score is RMSE divided by F1 score
    final_score = total_rmse / (f1_score + 1e-6)  # Avoid division by zero
    
    return f1_score, total_rmse, final_score


if __name__ == "__main__":
    perf_pred = 500
    lat = np.log(1e7 / (perf_pred + 1)) / 2
    perf = (1e7/np.exp(2*lat))-1
    print(perf_pred,perf,lat)