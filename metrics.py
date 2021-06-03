import numpy as np
import torch

def calc_evaluate(predict, mask):
    predict_class = (predict >= 0.5).float()
    predict_inverse = (predict_class == 0).float()
    mask_class = (mask >= 0.5).float()
    mask_inverse = (mask_class == 0).float()
    
    # calculate the value of TP, FP, TN and FN
    tp = predict_class.mul(mask_class).sum().tolist()
    fp = predict_class.mul(mask_inverse).sum().tolist()
    tn = predict_inverse.mul(mask_inverse).sum().tolist()
    fn = predict_inverse.mul(mask_class).sum().tolist()
    if tp == 0:
        tp = 1.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    iou = tp / (tp + fp + fn)
    # f1 == dice
    f1 = 2 * precision * recall / (precision + recall)
    f2 = 5 * precision * recall / (4 * precision + recall)
    accuracy_total = (tp + tn) / (tp + fp + tn + fn)
    
    res = [precision, recall, iou, f1, f2, accuracy_total]
    return res