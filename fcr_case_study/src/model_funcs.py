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
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from config import SEED
from xgboost import XGBClassifier


def get_RFC_model(model_params):
    criterions=["gini", "entropy", "log_loss"]
    max_features=["sqrt", "log2",None]
    class_weight=["balanced_subsample","balanced",None]
    return RandomForestClassifier(
        n_estimators=int(model_params["n_estimators"]),
        # max_depth=int(np.ceil(model_params["max_depth"])),
        criterion=criterions[int(np.trunc(model_params["criterion"]))],
        max_features=max_features[int(np.trunc(model_params["max_features"]))],
        class_weight=class_weight[int(np.trunc(model_params["class_weight"]))],
        random_state=SEED,
    )


def get_BRFC_model(model_params):
    criterions=["gini", "entropy"]
    max_features=["sqrt", "log2"]
    class_weight=["balanced_subsample","balanced",None]
    return BalancedRandomForestClassifier(
        n_estimators=int(model_params["n_estimators"]),
        # max_depth=int(np.ceil(model_params["max_depth"])),
        criterion=criterions[int(np.trunc(model_params["criterion"]))],
        max_features=max_features[int(np.trunc(model_params["max_features"]))],
        class_weight=class_weight[int(np.trunc(model_params["class_weight"]))],
        random_state=SEED,
    )


def get_SVC_model(model_params):
    return SVC(C=10 ** model_params["C"], class_weight="balanced", random_state=SEED)


def get_kNN_model(model_params):
    if model_params["weights"] >= 0.5:
        weights = "uniform"
    else:
        weights = "distance"
    return KNeighborsClassifier(
        n_neighbors=int(np.ceil(model_params["n_neighbors"])), weights=weights
    )


def get_XGboost_model(model_params):
    # booster=["gbtree", "gblinear", "dart"]
    return XGBClassifier(
        n_estimators=int(model_params["n_estimators"]),
        max_depth=int(np.ceil(model_params["max_depth"])),
        learning_rate=10 ** -model_params["learning_rate"],
        # booster=booster[int(np.trunc(model_params["booster"]))],
        random_state=SEED,
        verbosity=0
    )


class ThresholdedModel:
    latest_thresholds = (
        []
    )  # dirty method to get model thresholds of best cv evaluation
    # cant be used when doing multithreading
    best_score = np.finfo(np.float32).min

    def __init__(self, pipeline) -> None:  
        self.threshold = 0.5  
        self.model = pipeline

    def predict(self, X):
        return_values = np.where(self.model.predict_proba(X)[:, 1] >= self.threshold,1,0)
        return return_values 
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def set_threshold(self, threshold):
        self.threshold = threshold


def build_model(model_func, model_params, scaler_func):
    return ThresholdedModel(
        Pipeline(
            steps=(
                ("scaler", scaler_func()),
                (
                    "model",
                    model_func(model_params),
                ),
            )
        ),
    )
