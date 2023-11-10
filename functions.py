import numpy as np
from tqdm import tqdm
import torch

import json

BASELINE = ""

def load_baseline():
    # loads the baseline json file
    with open(BASELINE, 'r') as file:
        baseline = json.load(file)
        
    return baseline


def prediction_matrix(target, prediction):
    """ Create a prediction matrix using the target and prediction logits """
    predict_matrix = np.zeros((2,2))
    for t, p in zip(target, prediction):
        predict_matrix[int(t), int(p)] += 1

    return predict_matrix


def calculate_roc(target, prediction_prob, baseline):
    """ Create a prediction matrix using the target and prediction logits """
    roc_points = []
        
    for thresh in tqdm(np.unique(prediction_prob), desc="calculating performances", leave=False):
        pred_matrix = prediction_matrix(target, (prediction_prob >= thresh))
        fpr = (pred_matrix[0,1])/(pred_matrix[0,:]).sum()
        tpr = (pred_matrix[1,1])/(pred_matrix[1,:]).sum()
        roc_points.append(fpr, tpr)

    roc_points.sort(key=lambda x: x[0])

    # devide the points into bins and compare against baseline.
    rel_roc = []
    index = 0
    last_TPR = 0
    for i, v in enumerate(np.linspace(0, 1, 1000)):
        # get TPR
        while (index+1 < len(roc_points) # list must not run out
               and roc_points[index+1][0] <= v): # We need o find the TPR value near the closest FPR valie
           index += 1
           last_TPR = roc_points[index][1]

        rel_roc.append((v, last_TPR - baseline["roc"][i][1]))

    rel_auc = calculate_auc(roc_points)/baseline["AUC_ROC"]
    return rel_auc, rel_roc


def calculate_auc(roc_points):
    auc = 0
    prev_fpr = 0
    for tpr, fpr in roc_points:
        auc += (fpr - prev_fpr) * tpr
        prev_fpr = fpr

    return auc


def store_results(target, predict, fname, k=100):
    baseline = load_baseline()

    k_value = np.partition(predict, -k)[-k]
    
    auc, roc = calculate_roc(target, predict, baseline)
    pred_matrix = prediction_matrix(target, (predict >= k_value))
    
    results = {
        "rel_p@k": (pred_matrix[1,1])/(pred_matrix[:,1]).sum()/baseline["rel_p@k"],
        "rel_ra": auc,
        "rel_roc": roc
    }
    
    try:
        with open(fname, 'r') as file:
            prev_results = json.load(file)
    except FileNotFoundError:
            prev_results = []
    
    prev_results.append(results)
    
    with open(fname, 'w') as file:
        json.dump(prev_results, file)
    
    print("run", len(prev_results), " p@k:", "%.2f" % results["rel_p@k"])


def scramble_data(x, p_m):
    mask = torch.rand(x.shape) < p_m
    x_bar = x[torch.randperm(x.shape[0])]

    x_tilde = x*(~mask) + x_bar*mask
    mask_new = (x != x_tilde)

    return mask_new.float(), x_tilde
