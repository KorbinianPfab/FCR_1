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

import random
from typing import Any
import numpy as np
from metrics import (
    accuracy_score,
    f10_score,
    constrained_volume_reduction,
    constrained_AUC,
    volume_reduction_at_recall,
    precision_recall_auc
)
from sklearn.metrics import roc_auc_score
from utils import get_proper_threshold, log_metrics
from datasets import *
from config import SEED, TARGET_SLIP, TARGET_VOLUME_REDUCTION, AUTOML_TRAINING_TIME


import mlflow
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import make_scorer

from sklearn.model_selection import StratifiedKFold
from model_funcs import ThresholdedModel


random.seed(SEED)
np.random.seed(SEED)


mlflow.set_tracking_uri("http://localhost:5000")

class MetricWrapper:
    def __init__(self,scorer_func) -> None:
        self.scorer_func=scorer_func

    def __call__(self, y_true,y_pred) -> Any:
        y_pred=y_pred[:,1]
        return self.scorer_func(y_true,y_pred)


def get_automl_model(scorer_func):
    if scorer_func is None:
        scorer=None
    else:
        metric_wrapper=MetricWrapper(scorer_func)
        scorer = make_scorer(
            name=scorer_func.__name__,
            score_func=metric_wrapper,
            optimum=1,
            greater_is_better=True,
            needs_proba=True,
            needs_threshold=False,
        )
    return ThresholdedModel(
                    AutoSklearnClassifier(
                        AUTOML_TRAINING_TIME,
                        memory_limit=32000,
                        metric=scorer,
                        seed=SEED,
                    )
                )

def evaluate_autosklearn_model(scorer_func):
    if scorer_func is None:
        scorer_name="Default"
    else:
        scorer_name=scorer_func.__name__
    ys_thresholds=[]
    vrs_thresholds=[]
    with mlflow.start_run(
        run_name=f"AutoML_'DefaultThreshold_{scorer_name}_{SEED}"
    ):
        random.seed(SEED)
        np.random.seed(SEED)
        mlflow.set_tags(
            dict(
                AdaptThreshold=False,
                TargetSlip=TARGET_SLIP,
                TargetVolumeReduction=TARGET_VOLUME_REDUCTION,
                Seed=SEED,
                TargetFunction=str(scorer_func),
            )
        )
        skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
        for train, test in skf.split(X_hp, y_hp):
            model = get_automl_model(scorer_func)
            model = model.fit(X_hp[train], y_hp[train])
            ys_thresholds.append(
                get_proper_threshold(X_hp[test], y_hp[test], model, precision_recall_auc)
            )
            vrs_thresholds.append(
                get_proper_threshold(X_hp[test], y_hp[test], model, None)
            )
        mean_threshold = np.mean(vrs_thresholds)
        std_threshold = np.std(vrs_thresholds)
        selected_threshold = mean_threshold# - std_threshold
        if selected_threshold < 0:
            selected_threshold = 0
        model = get_automl_model(scorer_func)
        model.fit(X_hp, y_hp)
        log_metrics(X_test, y_test, model, name="Test", print_out=True)
        for idx, eval_slice in enumerate(eval_slices):
            log_metrics(eval_slice[0], eval_slice[1], model, name=f"Eval{idx}")
        print("Mean: ", mean_threshold)
        print("Std: ", std_threshold)
        print("Set_value: ", selected_threshold)
    with mlflow.start_run(
        run_name=f"AutoML_'AdaptedThreshold_VRS_{scorer_name}_{SEED}"
    ):
        model.set_threshold(selected_threshold)
        log_metrics(X_test, y_test, model, name="Test", print_out=True)
        for idx, eval_slice in enumerate(eval_slices):
            log_metrics(eval_slice[0], eval_slice[1], model, name=f"Eval{idx}")
        mlflow.log_params(
            dict(
                MeanThreshold=mean_threshold,
                StdThreshold=std_threshold,
                SetThreshold=selected_threshold,
            )
        )
    with mlflow.start_run(
        run_name=f"AutoML_'AdaptedThreshold_YS_{scorer_name}_{SEED}"
    ):
        mean_threshold = np.mean(ys_thresholds)
        std_threshold = np.std(ys_thresholds)
        selected_threshold = mean_threshold
        print("Mean: ", mean_threshold)
        print("Std: ", std_threshold)
        print("Set_value: ", selected_threshold)
        model.set_threshold(selected_threshold)
        log_metrics(X_test, y_test, model, name="Test", print_out=True)
        for idx, eval_slice in enumerate(eval_slices):
            log_metrics(eval_slice[0], eval_slice[1], model, name=f"Eval{idx}")
        mlflow.log_params(
            dict(
                MeanThreshold=mean_threshold,
                StdThreshold=std_threshold,
                SetThreshold=selected_threshold,
            )
        )

if __name__== '__main__':
    evaluate_autosklearn_model(None)
