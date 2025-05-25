import os
import numpy as np
import torch
import omegaconf
import lightning as pl
import json
from experiments.utils.model_definitions.vision_automodel_wrapper import VisionLayerwiseAutoModelWrapper, VisionModelSpecifications
from experiments.utils.dataloaders.vision_dataloader import prepare_datasets, prepare_dataloader, validation_imagenet_transform
from experiments.utils.dataloaders.convert_to_embeddings import convert_image_dataset_to_embeddings
from experiments.utils.misc.optimal_batch_size import find_optimal_batch_size
from experiments.utils.dataloaders.convert_to_embeddings import convert_image_dataset_to_embeddings
from experiments.utils.model_definitions.probe.LinearProbe import LinearModel

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers.utils import logging
logging.set_verbosity_error()

torch.set_float32_matmul_precision('medium')

def filter_dataset(dataset, classes):
    # set random seed
    np.random.seed(42)
    
    # limit to 1000 samples per class
    subsampled_indices = []
    for class_idx in classes:
        class_indices = [i for i, x in enumerate(dataset.targets) if x == class_idx]

        if len(class_indices) > 200:
            subsampled_indices.extend(np.random.choice(class_indices, size=100, replace=False))
        else:
            subsampled_indices.extend(class_indices)

    return torch.utils.data.Subset(dataset, subsampled_indices)

DATASET_NAME = "imagenet100"

image_transform = validation_imagenet_transform()
train_dataset = prepare_datasets(
    dataset=DATASET_NAME,
    transform=image_transform,
    train_data_path="/home/AD/ofsk222/Research/exploration/information_plane/experiments/datasets/imagenet100/train",
    number_of_samples=-1
)
val_dataset = prepare_datasets(
    dataset=DATASET_NAME,
    transform=image_transform,
    train_data_path="/home/AD/ofsk222/Research/exploration/information_plane/experiments/datasets/imagenet100/val",
    number_of_samples=-1
)

models_to_try = [
    # # Base models
    #VisionModelSpecifications(model_family="dinov2", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="mae", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="clip", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="vit", model_size="base", revision="main"),

    # Small models
    #VisionModelSpecifications(model_family="beit", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="dinov2-register", model_size="small", revision="main"),
    
    # # Large models
    VisionModelSpecifications(model_family="clip", model_size="large", revision="main"),
    VisionModelSpecifications(model_family="dinov2", model_size="large", revision="main"),
    VisionModelSpecifications(model_family="mae", model_size="large", revision="main"),
    VisionModelSpecifications(model_family="vit", model_size="large", revision="main"),
    VisionModelSpecifications(model_family="aim", model_size="large", revision="main"),
    VisionModelSpecifications(model_family="aimv2", model_size="large", revision="main"),

    VisionModelSpecifications(model_family="beit", model_size="large", revision="main"),

    # # Huge/giant models
    # VisionModelSpecifications(model_family="dinov2", model_size="giant", revision="main"),
    # VisionModelSpecifications(model_family="i-jepa", model_size="imagenet1k", revision="main"),
    #VisionModelSpecifications(model_family="i-jepa", model_size="imagenet21k", revision="main"),
    # VisionModelSpecifications(model_family="mae", model_size="huge", revision="main"),
    #VisionModelSpecifications(model_family="vit", model_size="huge", revision="main"),
    #VisionModelSpecifications(model_family="aim", model_size="huge", revision="main"),
    #VisionModelSpecifications(model_family="aim", model_size="1B", revision="main"),
    #VisionModelSpecifications(model_family="aim", model_size="3B", revision="main"),
]   

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--layer_window', type=int, default=0,
                   help='Window size for layer averaging')
parser.add_argument('--eval_layer', type=int, default=24,
                   help='Evaluation layer')
parser.add_argument('--model-selection-idx', type=int, default=5,
                   help='Model selection index')

args = parser.parse_args()

# make model
cfg = omegaconf.OmegaConf.create({
    "data": {
        "num_classes": 100,
    },
    "optimizer": {
        "weight_decay": 1e-4,
        "name": "adam",
        "lr": 1e-3,
    },
    "max_epochs": 1,
    "eval_layer": args.eval_layer,
    "layer_window": args.layer_window
})

for model_spec in [models_to_try[args.model_selection_idx]]:
    try:
        layer_range = (max(0, args.eval_layer - args.layer_window), args.eval_layer)
        save_path = f"vision_results/{model_spec.model_family}/{model_spec.model_size}/{DATASET_NAME}/attention_probe/layers-{layer_range[0]}-to-{layer_range[1]}.json"
        
        # if file already exists, skip
        if os.path.exists(save_path):
            print(f"Skipping {model_spec} because it already exists")
            continue

        backbone = VisionLayerwiseAutoModelWrapper(model_specs=model_spec, evaluation_layer_idx=cfg.eval_layer)
        probe = LinearModel(cfg=cfg, backbone=backbone).to(backbone._get_first_layer_device())


        optimal_batch_size = 128 #find_optimal_batch_size(probe, number_of_samples=1e6, batch_size=512, device=probe.device)
        
        print(model_spec)
        print(f"Optimal batch size: {optimal_batch_size}")

        train_dataloader = prepare_dataloader(train_dataset, batch_size=optimal_batch_size, num_workers=32, shuffle=True)
        val_dataloader = prepare_dataloader(val_dataset, batch_size=optimal_batch_size, num_workers=32, shuffle=False)

        # train model
        trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                              logger=False, devices=1,  precision='16-mixed')

        trainer.fit(probe, train_dataloader, val_dataloader)

        # save accuracies
        accuracies = dict(probe.trainer.callback_metrics)
        accuracies = {k: v.item() for k, v in accuracies.items()}

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(accuracies, f)

        del backbone
        del probe
        torch.cuda.empty_cache()
    except Exception as e:
        #raise e
        print(f"Error with {model_spec}")
        print(e)
        raise e
