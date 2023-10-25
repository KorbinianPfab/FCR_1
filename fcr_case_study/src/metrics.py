# LEGAL NOTICE © Siemens Aktiengesellschaft All rights reserved. 
# This software code is owned by Siemens Aktiengesellschaft and 
# is protected by the copyright laws as well as patent laws of 
# the United States and other countries, and by international 
# treaty provisions. No rights, whether by implication, by 
# estoppel or otherwise, are granted to you in connection with 
# the Software unless explicitly allowed hereunder. The only 
# permitted use of the Software is for non-commercial testing
# purposes in connection with the respective paper 'Towards 
# Improved Research Methodologies for Industrial AI: A Critical
# Examination using Automated Optical Inspection False Call 
# Reduction as a Case Study' by Siemens Aktiengesellschaft 
# submitted in connection with Engineering Applications of 
# Artificial Intelligence, Associations / Publisher - Springer.
# Any use outside the aforementioned scope shall be prohibited and
# may result in criminal and/or civil prosecution. No title to or
# ownership of the Software is transferred to you. Siemens 
# Aktiengesellschaft retains full and complete title to the 
# Software and all intellectual property rights therein. This
# Software is provided as is without warranty of any kind, either
# expressed or implied, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular 
# purpose. The entire risk as to the quality and performance of 
# the Software is with you. Should the Software prove defective,
# you assume the cost of all necessary servicing, repair or 
# correction. In no event will Siemens Aktiengesellschaft be 
# liable to you for any damages, including any lost profits, 
# lost savings or other incidental or consequential damages 
# arising out of the use or inability to use the Software. 
# By using the Software, you agree to be bound by the terms of
# this Legal Notice. 

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_curve,
    f1_score,
)
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.metrics import precision_recall_curve

from config import TARGET_SLIP, TARGET_VOLUME_REDUCTION


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)


def volume_reduction(y_true, y_pred):
    return recall_score(y_true, y_pred)


def f10_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=10)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision_recall_auc(y_true, y_pred):
    precision, recall, threshold = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


def constrained_AUC(y_true, y_pred):
    target_recall = 1 - TARGET_SLIP
    target_volume = 1 - TARGET_VOLUME_REDUCTION
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred)
    # thus false negatives
    #     are given by tps[-1] - tps
    fns = tps[-1] - tps
    # (thus true negatives are given by
    #     fps[-1] - fps).
    tns = fps[-1] - fps
    calc_business_metrics = lambda fp, tp, fn, tn: (tp / (tp + fn), tn / (tn + fp))
    recall_rates, volume_reductions = calc_business_metrics(fps, tps, fns, tns)
    recall_rates = np.append(recall_rates[::-1], 0)
    volume_reductions = np.append(volume_reductions[::-1], max(volume_reductions))
    auc_all = auc(volume_reductions, recall_rates)
    # return auc_all
    mask = recall_rates <= target_recall
    # idx_b = np.argmax(np.cumsum(mask))
    idx_b = np.argmax(mask)
    idx_a = idx_b - 1
    xp = [recall_rates[idx_a], recall_rates[idx_b]]
    fp = [volume_reductions[idx_a], volume_reductions[idx_b]]
    y_c = np.interp(x=target_recall, xp=xp, fp=fp)  # volume reduction at target slip

    mask = volume_reductions >= target_volume
    idx_e = np.argmax(mask)
    idx_d = idx_e - 1
    fp = [recall_rates[idx_d], recall_rates[idx_e]]
    xp = [volume_reductions[idx_d], volume_reductions[idx_e]]
    x_f = np.interp(x=target_volume, xp=xp, fp=fp)  # slip at target volume reduction

    alt_volume_reductions = (
        list(volume_reductions[volume_reductions < target_volume])
        + [target_volume] * 2
        + [y_c]
        + list(volume_reductions[recall_rates < target_recall])
    )
    alt_recall_rates = (
        list(recall_rates[volume_reductions < target_volume])
        + [x_f]
        + [target_recall] * 2
        + list(recall_rates[recall_rates < target_recall])
    )
    dx = np.diff(alt_volume_reductions)
    if np.all(dx >= 0):  # np.all(dx >= 0) or
        auc_alt = auc(alt_volume_reductions, alt_recall_rates)
        if auc_all > auc_alt:  # case when target area is met
            if (
                sum(
                    [
                        element[0] >= target_volume and element[1] >= target_recall
                        for element in zip(volume_reductions, recall_rates)
                    ]
                )
                > 0
            ):
                return (auc_all - auc_alt) / (TARGET_SLIP * TARGET_VOLUME_REDUCTION)
            else:
                return 0
    alt_volume_reductions = [min(vr, target_volume) for vr in volume_reductions]
    alt_recall_rates = [min(vr, target_recall) for vr in recall_rates]
    return (
        auc(alt_volume_reductions, alt_recall_rates) - (target_recall * target_volume)
    ) / (target_recall * target_volume)


def volume_reduction_at_recall(
    y_true, y_pred_proba, print_out=False, return_threshold=False
):
    target_recall = 1 - TARGET_SLIP
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred_proba)
    fns = tps[-1] - tps
    tns = fps[-1] - fps
    calc_business_metrics = lambda fp, tp, fn, tn: tp / (tp + fn)
    recall_rates = calc_business_metrics(fps, tps, fns, tns)
    threshold = thresholds[np.argmax(recall_rates >= target_recall)]
    y_pred = np.where(y_pred_proba >= threshold, 1, 0)
    if print_out:
        print("Perfect Threshold: ", threshold)
    if return_threshold:
        return recall_score(y_true, y_pred, pos_label=0), threshold
    return recall_score(y_true, y_pred, pos_label=0)


def youden_index(y_true, y_pred_proba, print_out=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba, drop_intermediate=False)
    idx = np.argmax(tpr - fpr)
    threshold = thresholds[idx]
    if print_out:
        print("Perfect Threshold: ", threshold)
    return tpr[idx] - fpr[idx]


def constrained_volume_reduction(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    r_score = tp / (tp + fn) 
    if r_score < 1 - TARGET_SLIP:
        return r_score - (1 - TARGET_SLIP)
    return tn / (tn + fp)
