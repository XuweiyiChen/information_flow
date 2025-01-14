import logging
import time
from typing import Any, Callable, List, Literal, Type, Dict, Union
import gc

import tqdm
import numpy as np
import torch
from transformers import BatchFeature, AutoModel, AutoImageProcessor, AutoConfig, CLIPVisionModel, CLIPVisionConfig
from transformers.models import dinov2
from torch.utils.data import DataLoader
import timm

from .base_automodel_wrapper import BaseModelSpecifications, BaseLayerwiseAutoModelWrapper
from ..misc.optimal_batch_size import find_optimal_batch_size
from .jepa.JepaEncoder import load_jepa_encoder

model_name_to_sizes = {
    'sam': ['base'],
    'vit_augreg': ['base'],
    'vit': ['base', 'large', 'huge'],
    'dinov1': ['base'],
    'dinov2': ['small', 'base', 'large', 'giant'],
    'dinov2-register': ['small', 'base', 'large', 'giant'],
    'mae': ['base', 'large', 'huge'],
    'deit': ['base'],
    'clip': ['base', 'large'],
    'i-jepa': ['imagenet1k', 'imagenet21k'],
}
model_types = list(model_name_to_sizes.keys())

def get_model_path(name, size):
    assert name in model_types, f"Invalid model type {name}, valid types: {model_types}"
    assert size in model_name_to_sizes[name], \
        f"Invalid size {size} for model type {name}, valid sizes: {model_name_to_sizes[name]}"
    
    if name == 'vit':
        patch_size = 16 if size != 'huge' else 14
        dataset = "-in21k" if size == 'huge' else ""
        return f"google/vit-{size}-patch{patch_size}-224{dataset}"
    elif name == 'dinov1':
        return f'facebook/dino-vitb16'
    elif name == 'dinov2':
        return f'facebook/dinov2-{size}'
    elif name == 'dinov2-register':
        return f'timm/vit_{size}_patch14_reg4_dinov2.lvd142m'
    elif name == 'mae':
        return f"facebook/vit-mae-{size}"
    elif name == 'sam':
        return "facebook/sam-vit-base"
    elif name == 'vit_augreg':
        return "timm/vit_base_patch16_224.augreg_in21k"
    elif name == 'deit':
        return "facebook/deit-base-distilled-patch16-224"
    elif name == 'clip':
        if size == 'base':
            return "openai/clip-vit-base-patch16"
        elif size == 'large':
            return "openai/clip-vit-large-patch14"
    elif name == 'i-jepa':
        return ""

def update_config(config, model_specs):
    if model_specs.model_family == 'mae':
        config.mask_ratio = 0.

    return config

def get_model_and_config_classes(model_specs):
    if model_specs.model_family == 'clip':
        return CLIPVisionModel, CLIPVisionConfig
    else:
        return AutoModel, AutoConfig



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
        if self._is_timm_model():
           data_config = timm.data.resolve_model_data_config(self.model)
           self.image_processor = timm.data.create_transform(**data_config, is_training=False)
        elif self.model_specs.model_family == 'i-jepa':
            self.image_processor = lambda x: x
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)

    def process_inputs(self, inputs):
        if self._is_timm_model():
            return self.image_processor(inputs)
        elif self.model_specs.model_family == 'i-jepa':
            return inputs
        else:
            return self.image_processor(inputs, return_tensors="pt")

    def setup_model(self):
        if self._is_timm_model():
            self.setup_timm_model()
        elif self.model_specs.model_family == 'i-jepa':
            self.setup_jepa_model()
        else:
            self.setup_huggingface_model()

    def setup_jepa_model(self):
        self.model = load_jepa_encoder(self.model_specs.model_size)

    def setup_huggingface_model(self):
        MODEL_CLASS, CONFIG_CLASS = get_model_and_config_classes(self.model_specs)
        self.config = CONFIG_CLASS.from_pretrained(self.model_path, 
                                            revision=self.model_specs.revision,
                                            output_hidden_states=True)
        self.config = update_config(self.config, self.model_specs)
        #self.num_layers = self.config.num_hidden_layers + 1 
        #self.update_evaluation_layer(self.evaluation_layer_idx)
        #self.config.num_hidden_layers = self.evaluation_layer_idx

        FROM_PRETRAINED_KWARGS = {
            'revision': self.model_specs.revision,
            'config': self.config,
            'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            'device_map': self.device_map if False else None
        }

        self.model = MODEL_CLASS.from_pretrained(self.model_path, **FROM_PRETRAINED_KWARGS).eval()

        if FROM_PRETRAINED_KWARGS['device_map'] is None:
            self.model.to("cuda")

    def setup_timm_model(self):
        base_model_path = self.model_path.split('/')[1]
        self.model = timm.create_model(base_model_path, pretrained=True, num_classes=0)
        self.model = self.model.eval().cuda()

    def __call__(self, **kwargs):
        if 'timm' in self.model_path:
            return {
                'hidden_states': self.model.forward_intermediates(**kwargs, intermediates_only=True, output_fmt='NLC')
            }
        else:
            return self.forward(**kwargs)
        
    def prepare_inputs(self, batch):
        batch_idx, images, labels = batch
        batch_size = len(batch_idx)
            
        if isinstance(images, BatchFeature):
            inputs = images.to("cuda")
            inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(self.dtype)
        elif 'timm' in self.model_path or self.model_specs.model_family == 'i-jepa':
            inputs = {
                "x": images.to("cuda").to(self.dtype)
            }
        else:
            inputs = {
                "pixel_values": images.to("cuda").to(self.dtype)
            }
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        return inputs
    
    def _is_timm_model(self):
        return 'timm' in self.model_path