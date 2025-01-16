from typing import Any, Callable, List, Literal, Type, Dict, Union
import torch
import tqdm
import numpy as np

import repitl.matrix_itl as itl
import repitl.difference_of_entropies as dent

from . import metric_functions as mf
from ..model_definitions.base_automodel_wrapper import BaseModelSpecifications

DISABLE_TQDM = False

metric_name_to_function = {
    'entropy': mf.compute_entropy,
    'lidar': mf.compute_lidar,
    'dime': mf.compute_dime,
    'infonce': mf.compute_infonce,
    'curvature': mf.compute_curvature,
    'intrinsic_dimension': mf.compute_intrinsic_dimension,
}

class EvaluationMetricSpecifications:
    def __init__(
        self, 
        evaluation_metric, 
        num_samples = 1000, 
        alpha = 1, 
        normalizations = ['maxEntropy', 'raw', 'logN', 'logNlogD', 'logD'],
        curvature_k = 1
    ):
        self.evaluation_metric = evaluation_metric
        self.num_samples = num_samples

        
        if self.evaluation_metric == 'sentence-entropy':
            self.granularity = 'sentence'
            self.evaluation_metric = 'entropy'
        elif self.evaluation_metric == 'dataset-entropy':
            self.granularity = 'dataset'
            self.evaluation_metric = 'entropy'
        else:
            self.granularity = None

        # for matrix-based metrics (LIDAR, DIME, entropy)
        self.normalizations = normalizations
        self.alpha = alpha

        # for curvature
        self.curvature_k = curvature_k
        
        self.do_checks()

    def do_checks(self):
        assert self.evaluation_metric in metric_name_to_function.keys()
        assert self.granularity in ['sentence', 'dataset', None]

        assert self.alpha > 0
        assert self.num_samples > 0
        assert self.curvature_k > 0 and isinstance(self.curvature_k, int)

    def __str__(self):
        return f"Metric: {self.evaluation_metric}"

def compute_per_forward_pass(model, dataloader, num_examples,compute_function, should_average_over_layers=False, **kwargs):
    """
    Compute a metric for each forward pass through the model.

    Args:
        model (torch.nn.Module): The model to use for forward passes.
        dataloader (torch.utils.data.DataLoader): The dataloader providing batches.
        compute_function (callable): The function to compute the metric.
        **kwargs: Additional keyword arguments to pass to compute_function.

    Returns:
        dict: A dictionary of computed metrics, averaged over all samples.
    """
    results = {}
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, total=num_examples, disable=DISABLE_TQDM, desc="Processing batches"):
            batch = model.prepare_inputs(batch)
            outputs = model(**batch)
            
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
            elif isinstance(outputs, dict) and 'hidden_states' in outputs:
                hidden_states = outputs['hidden_states']
            else:
                hidden_states = outputs

            for sample_idx in range(len(hidden_states[0])):
                if hasattr(batch, 'attention_mask'):
                    # ignore padding tokens
                    pad_idx = batch['attention_mask'][sample_idx] == 0
                else:
                    pad_idx = None

                sample_hidden_states = [
                    mf.normalize(layer_states[sample_idx][~pad_idx]) if pad_idx is not None
                    else mf.normalize(layer_states[sample_idx])
                    for layer_states in hidden_states
                ]
                sample_hidden_states = torch.stack(sample_hidden_states) # L x NUM_TOKENS x D

                sample_result = compute_function(sample_hidden_states, **kwargs)
                for norm, values in sample_result.items():
                    if norm not in results:
                        results[norm] = []
                    results[norm].append(values)

    if should_average_over_layers:
        return {norm: np.array(values).mean(axis=0) for norm, values in results.items()}
    else:
        return {norm: np.array(values) for norm, values in results.items()}

def compute_on_concatenated_passes(model, dataloader, num_examples, compute_function, **kwargs):
    """
    Compute a metric on concatenated hidden states from multiple forward passes.

    Args:
        model (torch.nn.Module): The model to use for forward passes.
        dataloader (torch.utils.data.DataLoader): The dataloader providing batches.
        num_examples (int): The number of examples in the dataloader, can be tough to get directly from the dataloader in some cases. Only used for tqdm.
        compute_function (callable): The function to compute the metric.
        **kwargs: Additional keyword arguments to pass to compute_function.

    Returns:
        dict: A dictionary of computed metrics.
    """
    all_hidden_states = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, total=num_examples, disable=DISABLE_TQDM):
            if not isinstance(batch, tuple) and not isinstance(batch, list):
                batch = (batch,)
            
            batch_hidden_states = []
            for sub_batch in batch:
                sub_batch = model.prepare_inputs(sub_batch)
               
                outputs = model(**sub_batch)
                hidden_states = [mf.normalize(x.squeeze()) for x in outputs.hidden_states] # L x BS x NUM_TOKENS x D
                
                # uncomment for CLS token
                #layer_means = torch.stack([x[:, 0, :] for x in hidden_states]) # L x BS x D
                
                layer_means = torch.stack([torch.mean(x, dim=0) for x in hidden_states]) # L x BS x D
                if len(layer_means.shape) == 2:
                    layer_means = layer_means.unsqueeze(1) # L x BS x D

                batch_hidden_states.append(layer_means)
            
            all_hidden_states.append(torch.stack(batch_hidden_states)) # NUM_AUG x L x BS x D
 
    concatenated_states = torch.stack(all_hidden_states) # NUM_BATCHES x NUM_AUG x L x BS x D
    concatenated_states = concatenated_states.view(
        concatenated_states.shape[0] * concatenated_states.shape[3], 
        concatenated_states.shape[1],
        concatenated_states.shape[2],
        concatenated_states.shape[4]
    ) # NUM_SAMPLES x NUM_AUG x L x D
    concatenated_states = concatenated_states.permute(2, 0, 1, 3) # L x NUM_SAMPLES x NUM_AUG x D
    concatenated_states = concatenated_states.squeeze()
    return compute_function(concatenated_states, **kwargs)


def calculate_and_save_layerwise_metrics(
    model,
    dataloader,
    model_specs: BaseModelSpecifications,
    evaluation_metric_specs: EvaluationMetricSpecifications,
    dataloader_kwargs: Dict[str, Any],
):
    if evaluation_metric_specs.evaluation_metric == 'entropy':
        compute_func_kwargs = {
            'alpha': evaluation_metric_specs.alpha,
            'normalizations': evaluation_metric_specs.normalizations
        }
        forward_pass_func = compute_per_forward_pass if evaluation_metric_specs.granularity == 'sentence' else compute_on_concatenated_passes
  

    elif evaluation_metric_specs.evaluation_metric == 'curvature':
        compute_func_kwargs = {
            'k': evaluation_metric_specs.curvature_k
        }
        forward_pass_func = compute_per_forward_pass

    elif evaluation_metric_specs.evaluation_metric == 'lidar':
        compute_func_kwargs = {
            'alpha': evaluation_metric_specs.alpha,
            'normalizations': evaluation_metric_specs.normalizations,
        }
        forward_pass_func = compute_on_concatenated_passes

    elif evaluation_metric_specs.evaluation_metric == 'dime':
        compute_func_kwargs = {
            'alpha': evaluation_metric_specs.alpha,
            'normalizations': evaluation_metric_specs.normalizations,
        }
        forward_pass_func = compute_on_concatenated_passes

    elif evaluation_metric_specs.evaluation_metric == 'infonce':
        compute_func_kwargs = {
            'temperature': 0.1,
        }
        forward_pass_func = compute_on_concatenated_passes
    
    elif evaluation_metric_specs.evaluation_metric == 'intrinsic_dimension':
        compute_func_kwargs = {}
        forward_pass_func = compute_per_forward_pass

    compute_func = metric_name_to_function[evaluation_metric_specs.evaluation_metric]
    results = forward_pass_func(model, dataloader, dataloader_kwargs['num_samples'], compute_func, **compute_func_kwargs)

    from utils.misc.results_saving import save_results # here to avoid circular imports
    save_results(results, model_specs, evaluation_metric_specs, dataloader_kwargs)

    return results