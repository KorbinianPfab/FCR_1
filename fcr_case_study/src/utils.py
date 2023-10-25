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

import mlflow
import random
import numpy as np
import pandas as pd
from sklearn.metrics._ranking import _binary_clf_curve

from fcr_case_study.src.config import (
    TARGET_SLIP,
    TARGET_VOLUME_REDUCTION,
    BO_INIT_POINTS,
    BO_N_ITER,
)

import plotly.graph_objects as go
from metrics import (
    constrained_AUC,
    constrained_volume_reduction,
    f10_score,
    volume_reduction_at_recall,
    accuracy_score,
    accuracy,
    recall,
    youden_index,
    precision_recall_auc,
)
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    roc_curve,
    f1_score,
    roc_auc_score,
)

from config import SEED

random.seed(SEED)
np.random.seed(SEED)


def evaluate_model(X, y, model, metric):
    if metric in (
        constrained_AUC,
        volume_reduction_at_recall,
        youden_index,
        precision_recall_auc,
        roc_auc_score,
    ):
        y_pred = model.predict_proba(X)[:, 1]
    else:
        y_pred = model.predict(X)
    return metric(y, y_pred)


def log_metrics(X, y_true, model, name, print_out=False):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    vr_at_s, perfect_threshold = (
        volume_reduction_at_recall(y_true, y_pred_proba, return_threshold=True)
    )
    mlflow.log_metrics(
        metrics={
            f"Accuracy{name}": accuracy_score(y_true, y_pred),
            f"F1Score{name}": f1_score(y_true, y_pred),
            f"Recall{name}": recall_score(y_true, y_pred),
            f"VolumeReduction{name} ": recall_score(y_true, y_pred, pos_label=0),
            f"ConstrainedVolumeReduction{name}": constrained_volume_reduction(
                y_true, y_pred
            ),
            f"VolumeReductionAtRecall{name}": vr_at_s,
            f"PerfectThreshold{name}": perfect_threshold,
            f"ConstrainedAUC{name}": constrained_AUC(y_true, y_pred_proba),
            f"PrecisionRecallAUC{name}": precision_recall_auc(y_true, y_pred_proba),
            f"YoudenIndex{name}": youden_index(y_true, y_pred_proba),
        }
    )
    fig = create_confusion_matrix_plot(
        confusion_matrix(y_true, y_pred, normalize="true")
    )
    mlflow.log_figure(fig, f"confusionmatrix{name}.svg")
    mlflow.log_dict(fig.to_json(), f"confusionmatrixfig{name}.json")
    if print_out:
        print("Accuracy: ", accuracy_score(y_true, y_pred))
        print("Volume Reduction: ", recall_score(y_true, y_pred, pos_label=0))
        print("F1 Score: ", f1_score(y_true, y_pred))
        print("Recall: ", recall_score(y_true, y_pred))
        print(
            "Constrained Volume Reduction: ",
            constrained_volume_reduction(y_true, y_pred),
        )
        print(
            "Constrained AUC: ",
            constrained_AUC(y_true, y_pred_proba),
        )
        print(
            "Volume reduction at recall: ",
            volume_reduction_at_recall(y_true, y_pred_proba, True),
        )
        print("YoudenIndex: ", youden_index(y_true, y_pred_proba))
        print(confusion_matrix(y_true, y_pred))


def log_standard_models_attributes(adapted_threshold, metric, params, target):
    mlflow.set_tags(
        dict(
            AdaptThreshold=adapted_threshold,
            TargetSlip=TARGET_SLIP,
            TargetVolumeReduction=TARGET_VOLUME_REDUCTION,
            Seed=SEED,
            BoInitPoints=BO_INIT_POINTS,
            BoNIter=BO_N_ITER,
            TargetFunction=metric,
        )
    )
    mlflow.log_params(params=params)
    mlflow.log_metric("Target", target)


def create_confusion_matrix_plot(confusion_matrix):
    fig = go.Figure(go.Heatmap(z=confusion_matrix, x=["0", "1"], y=["0", "1"]))
    labels = [0, 1]
    annotations = []
    for i, row in enumerate(confusion_matrix):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": labels[j],
                    "y": labels[i],
                    "font": {"color": "white"},
                    "text": str(np.round(value * 100, 2)) + "%",
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False,
                }
            )
    fig.update_layout(
        height=450,
        width=450,
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label"),
        annotations=annotations,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def get_proper_threshold(X, y_true, model, metric_func):
    if metric_func in (accuracy, recall, f1_score, roc_auc_score, precision_recall_auc):
        fpr, tpr, thresholds = roc_curve(y_true, model.predict_proba(X)[:, 1])
        idx = np.argmax(tpr - fpr)
        return thresholds[idx]
    else:
        target_recall = 1 - TARGET_SLIP
        y_pred = model.predict_proba(X)[:, 1]
        fps, tps, thresholds = _binary_clf_curve(y_true, y_pred)
        # thus false negatives
        #     are given by tps[-1] - tps
        fns = tps[-1] - tps
        # (thus true negatives are given by
        #     fps[-1] - fps).
        tns = fps[-1] - fps
        calc_business_metrics = lambda fp, tp, fn, tn: tp / (tp + fn)
        recall_rates = calc_business_metrics(fps, tps, fns, tns)
        return thresholds[np.argmax(recall_rates >= target_recall)]

