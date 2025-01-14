import os
import pickle
from typing import Any, Callable, List, Literal, Type, Dict, Union

from ..metrics.metric_calling import EvaluationMetricSpecifications
from ..model_definitions.base_automodel_wrapper import BaseModelSpecifications

BASE_PATH = "/home/AD/ofsk222/Research/exploration/information_plane/experiments/large_results"

def construct_file_path(
        model_specs: BaseModelSpecifications, 
        evaluation_metric_specs: EvaluationMetricSpecifications, 
        dataloader_kwargs: Dict[str, Any]
):
    model_family = model_specs.model_family
    model_size = model_specs.model_size
    revision = model_specs.revision
    evaluation_metric = evaluation_metric_specs.evaluation_metric
    granularity = evaluation_metric_specs.granularity
    dataset = dataloader_kwargs['dataset_name']

    if evaluation_metric == 'entropy':
        evaluation_metric = f"{evaluation_metric}_{granularity}"

    return f"{BASE_PATH}/{model_family}/{model_size}/{revision}/metrics/{dataset}/{evaluation_metric}.pkl"

def save_results(
        results, 
        model_specs: BaseModelSpecifications, 
        evaluation_metric_specs: EvaluationMetricSpecifications, 
        dataloader_kwargs: Dict[str, Any]
):
    file_path = construct_file_path(model_specs, evaluation_metric_specs, dataloader_kwargs)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(results, f)

def check_if_results_exist(
        model_specs: BaseModelSpecifications, 
        evaluation_metric_specs: EvaluationMetricSpecifications, 
        dataloader_kwargs: Dict[str, Any]
):
    file_path = construct_file_path(model_specs, evaluation_metric_specs, dataloader_kwargs)
    return os.path.exists(file_path)


def load_results(
        model_specs: BaseModelSpecifications, 
        evaluation_metric_specs: EvaluationMetricSpecifications, 
        dataloader_kwargs: Dict[str, Any]
):
    file_path = construct_file_path(model_specs, evaluation_metric_specs, dataloader_kwargs)

    try:
        with open(file_path, "rb") as f:
            results = pickle.load(f)
        return results
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def load_results_for_model_and_revisions(model_family, model_size, revisions, evaluation_metrics):
    results = {}
    for revision in revisions:
        model_specs = BaseModelSpecifications(model_family, model_size, revision)
        for evaluation_metric in evaluation_metrics:
            evaluation_metric_specs = EvaluationMetricSpecifications(evaluation_metric)
            dataloader_kwargs = {'dataset_name': 'wikitext'}
            results[(revision, evaluation_metric)] = load_results(model_specs, evaluation_metric_specs, dataloader_kwargs)
    return results

def load_all_results():
    all_results = {}
    results_dir = BASE_PATH

    for model_family in os.listdir(results_dir):
        model_family_path = os.path.join(results_dir, model_family)
        if os.path.isdir(model_family_path):
            all_results[model_family] = {}
            
            for model_size in os.listdir(model_family_path):
                model_size_path = os.path.join(model_family_path, model_size)
                if os.path.isdir(model_size_path):
                    all_results[model_family][model_size] = {}
                    
                    for revision in os.listdir(model_size_path):
                        revision_path = os.path.join(model_size_path, revision)
                        if os.path.isdir(revision_path):
                            all_results[model_family][model_size][revision] = {}
                            
                            metrics_path = os.path.join(revision_path, "metrics")
                            if os.path.isdir(metrics_path):
                                for dataset in os.listdir(metrics_path):
                                    dataset_path = os.path.join(metrics_path, dataset)
                                    if os.path.isdir(dataset_path):
                                        all_results[model_family][model_size][revision][dataset] = {}
                                        
                                        for metric_file in os.listdir(dataset_path):
                                            if metric_file.endswith('.pkl'):
                                                metric_name = os.path.splitext(metric_file)[0]
                                                file_path = os.path.join(dataset_path, metric_file)
                                                with open(file_path, 'rb') as f:
                                                    metric_results = pickle.load(f)
                                                all_results[model_family][model_size][revision][dataset][metric_name] = metric_results

    return all_results