import logging
import time
from typing import Any, Callable, List, Literal, Type, Dict, Union
import gc

import tqdm
import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader

from .base_automodel_wrapper import BaseModelSpecifications, BaseLayerwiseAutoModelWrapper
from ..misc.optimal_batch_size import find_optimal_batch_size
from llm2vec import LLM2Vec

model_types = ['vit', 'dinov1', 'dinov2', 'mae']
def get_model_path(name, size):
    assert name in model_types, f"Invalid model type {name}, valid types: {model_types}"
    
    if name == 'vit':
        return "google/vit-base-patch16-224-in21k"
    elif name == 'dinov1':
        return 'facebook/dino-vitb16'
    elif name == 'dinov2':
        return 'facebook/dinov2-base'
    elif name == 'mae':
        return "facebook/vit-mae-base"



class VisionModelSpecifications(BaseModelSpecifications):
    def __init__(self, model_family, model_size, revision):
        super().__init__(model_family, model_size, revision)
        self.model_path_func = get_model_path

    def additional_checks(self):
        pass

class VisionLayerwiseAutoModelWrapper(BaseLayerwiseAutoModelWrapper):
    def __init__(self, 
                 model_specs: VisionModelSpecifications, 
                 device_map="auto", 
                 evaluation_layer_idx: int = -1):
        super().__init__(model_specs, device_map, evaluation_layer_idx)

    """
    FUNCTIONS FOR INITIALIZATION
    """
    def setup_input_processor(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)

    def setup_model(self):
        self.config = AutoConfig.from_pretrained(self.model_path, 
                                            revision=self.model_specs.revision,
                                            output_hidden_states=True)
        self.num_layers = self.config.num_hidden_layers + 1 
        self.update_evaluation_layer(self.evaluation_layer_idx)
        self.config.num_hidden_layers = self.evaluation_layer_idx

        FROM_PRETRAINED_KWARGS = {
            'revision': self.model_specs.revision,
            'config': self.config,
            'torch_dtype': torch.float32,
            #'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            'device_map': self.device_map if False else None
        }

        self.model = AutoModel.from_pretrained(self.model_path, **FROM_PRETRAINED_KWARGS).eval()

        if FROM_PRETRAINED_KWARGS['device_map'] is None:
            self.model.to("cuda")