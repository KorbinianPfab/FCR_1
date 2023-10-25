import random
from copy import deepcopy

import mlflow
import numpy as np

from bayes_opt import BayesianOptimization
from config import (
    BO_VERBOSITY,
    BO_INIT_POINTS,
    BO_N_ITER,
    SEED,
)
from imblearn.under_sampling import RandomUnderSampler
from param import *
from param import model_params
from sklearn.model_selection import StratifiedKFold
from fcr_case_study.src.utils import get_proper_threshold
from utils import evaluate_model, log_metrics, log_standard_models_attributes

from model_funcs import ThresholdedModel, build_model
from datasets import *

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

random.seed(SEED)
np.random.seed(SEED)


mlflow.set_tracking_uri("http://localhost:5000")


def bayesian_run(model_func, metric_func, **kwargs):
    thresholds = []
    sampler = RandomUnderSampler(
        sampling_strategy=1 / kwargs["sampling_ratio"], random_state=SEED
    )
    scaler_func = scaler_params["scaler_options"][int(kwargs["scaler_kind"])]
    model_params = deepcopy(kwargs)
    model_params.pop("sampling_ratio")
    model_params.pop("scaler_kind")
    model = build_model(model_func, model_params, scaler_func)
    skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    scores = []
    for train, test in skf.split(X_hp, y_hp):
        X_sampled, y_sampled = sampler.fit_resample(X_hp[train], y_hp[train])
        model = model.fit(X_sampled, y_sampled)
        score = evaluate_model(X_hp[test], y_hp[test], model, metric_func)
        thresholds.append(get_proper_threshold(X_hp[test], y_hp[test], model,metric_func))
        scores.append(score)
    avg_scores = np.average(scores)
    if avg_scores > ThresholdedModel.best_score:
        ThresholdedModel.best_score = avg_scores
        ThresholdedModel.latest_thresholds = thresholds
    return avg_scores


def hp_optimization(model_config, metric_func):
    def bayesian_run_wrapper(**kwargs):
        ThresholdedModel.latest_thresholds = []        
        ThresholdedModel.best_score = np.finfo(np.float32).min
        return bayesian_run(model_config["model_func"], metric_func, **kwargs)

    all_param_bounds = model_config["param_bounds"]
    all_param_bounds.update(scaler_params["param_bounds"])
    all_param_bounds.update(sampler_params["param_bounds"])
    optimizer = BayesianOptimization(
        f=bayesian_run_wrapper,
        pbounds=all_param_bounds,
        verbose=BO_VERBOSITY,
        random_state=SEED,
    )
    optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_N_ITER)
    print(
        "Best result: {}; f(x) = {}.".format(
            optimizer.max["params"], optimizer.max["target"]
        )
    )
    return optimizer.max["params"], optimizer.max["target"]


def evaluate_standard_models():
    for model_name in model_params.keys():
        for metric in metric_params.keys():
            with mlflow.start_run(
                run_name=f"{model_name}_{metric}_DefaultThreshold_{SEED}"
            ):
                random.seed(SEED)
                np.random.seed(SEED)
                
                print("\n", "\n", "-" * 100)
                params, target = hp_optimization(
                    model_params[model_name], metric_params[metric]
                )
                log_standard_models_attributes(False,metric,params,target)
                scaler_func = scaler_params["scaler_options"][
                    int(params["scaler_kind"])
                ]
                opt_params = deepcopy(params)
                opt_params.pop("sampling_ratio")
                # model_params.pop("sampler_kind")
                opt_params.pop("scaler_kind")
                model = build_model(
                    model_params[model_name]["model_func"], opt_params, scaler_func
                )
                sampler = RandomUnderSampler(
                    sampling_strategy=1 / params["sampling_ratio"], random_state=SEED
                )
                X_sampled, y_sampled = sampler.fit_resample(X_hp, y_hp)
                model.fit(X_sampled, y_sampled)
                print("Model: ", model_name)
                print("Target metric: ", metric)
                log_metrics(X_test, y_test, model, name="Test", print_out=True)
                for idx, eval_slice in enumerate(eval_slices):
                    log_metrics(eval_slice[0], eval_slice[1], model, name=f"Eval{idx}")
            with mlflow.start_run(
                run_name=f"{model_name}_{metric}_AdaptedThreshold_{SEED}"
            ):
                mean_threshold = np.mean(ThresholdedModel.latest_thresholds)
                std_threshold = np.std(ThresholdedModel.latest_thresholds)
                selected_threshold = mean_threshold# - std_threshold
                if selected_threshold < 0:
                    selected_threshold = 0
                print("Mean: ", mean_threshold)
                print("Std: ", std_threshold)
                print("Max: ", max(ThresholdedModel.latest_thresholds))
                print("Min: ", min(ThresholdedModel.latest_thresholds))
                print("Set_value: ", selected_threshold)
                print("\n","After setting threshold: ")
                mlflow.log_params(
                    dict(
                        MeanThreshold=mean_threshold,
                        StdThreshold=std_threshold,
                        SetThreshold=selected_threshold,
                    )
                )
                log_standard_models_attributes(True,metric,params,target)
                model.set_threshold(selected_threshold)
                log_metrics(X_test, y_test, model, name="Test", print_out=True)
                for idx, eval_slice in enumerate(eval_slices):
                    log_metrics(eval_slice[0], eval_slice[1], model, name=f"Eval{idx}")


if __name__ == "__main__":
    evaluate_standard_models()
