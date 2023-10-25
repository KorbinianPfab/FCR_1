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

from metrics import (
    accuracy,
    constrained_AUC,
    constrained_volume_reduction,
    f10_score,
    volume_reduction_at_recall,
    recall,
    youden_index,
    precision_recall_auc
)
from model_funcs import (
    get_BRFC_model,
    get_kNN_model,
    get_RFC_model,
    get_SVC_model,
    get_XGboost_model,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score,roc_auc_score

model_params = {
    "RFC": {
        "model_func": get_RFC_model,
        "param_bounds": {
            "n_estimators": [20, 400],
            "criterion": [0, 2.99],
            "max_features": [0, 1.99],
            "class_weight": [0, 2.99],
        },
    },
    "BRFC": {
        "model_func": get_BRFC_model,
        "param_bounds": {
            "n_estimators": [20, 400],
            "criterion": [0, 1.99],
            "max_features": [0, 1.99],
            "class_weight": [0, 2.99],
        },
    },
    "kNN": {
        "model_func": get_kNN_model,
        "param_bounds": {
            "n_neighbors": [1, 100],
            "weights": [0.0, 1.0],
        },
    },
    "XGBoost": {
        "model_func": get_XGboost_model,
        "param_bounds": {
            "n_estimators": [20, 600],
            "max_depth": [10, 40],
            "learning_rate": [-1, 3],
        },
    },
}

metric_params = {
    "PrecisionRecallAUC": precision_recall_auc,
    "ConstrainedAUC": constrained_AUC,
}


scaler_params = {
    "scaler_options": [StandardScaler, MinMaxScaler],
    "param_bounds": {"scaler_kind": [0, 1.999]},
}

sampler_params = {
    "param_bounds": {"sampling_ratio": [1, 40]}, 
}
