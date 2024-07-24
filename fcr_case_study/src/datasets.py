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

from config import SEED
import warnings
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

random.seed(SEED)
np.random.seed(SEED)


def load_and_prepare_data():
    df = pd.read_csv(
        "./fcr_case_study/data/anonymized_data/FCR_dataset.csv",
        index_col=0,
    ).drop(columns="timestamp")
    categorical_columns = [
        "meta_feat1",
        "meta_feat2",
        "meta_feat3",
        "meta_feat4",
        "inspection_type",
    ]
    for cat_column in categorical_columns:
        df[cat_column].values.reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False).fit(df[cat_column].values.reshape(-1, 1))
        df[
            [
                "_".join([cat_column, str(idx)])
                for idx in range(len(encoder.get_feature_names_out()))
            ]
        ] = encoder.transform(df[cat_column].values.reshape(-1, 1)).tolist()
        df.drop(columns=cat_column, inplace=True)
        df = df.copy()
    return split_datasets(df)


def split_datasets(df):
    df_len = len(df)
    X_all, y_all = df.drop(columns="class").values, df["class"].values
    X_1, y_1 = X_all[: int(df_len * 0.5)], y_all[: int(df_len * 0.5)]
    X_2, y_2 = X_all[int(df_len * 0.5) :], y_all[int(df_len * 0.5) :]
    X_hp, X_test, y_hp, y_test = train_test_split(
        X_1, y_1, test_size=0.2, stratify=y_1, random_state=SEED
    )
    X2_len = len(X_2)
    eval_slices = []
    for i in range(5):
        eval_slices.append(
            (
                X_2[int(i * 0.2 * X2_len) : int((i + 1) * 0.2 * X2_len)],
                y_2[int(i * 0.2 * X2_len) : int((i + 1) * 0.2 * X2_len)],
            )
        )
    return X_hp, y_hp, X_test, y_test, eval_slices


X_hp, y_hp, X_test, y_test, eval_slices = load_and_prepare_data()
