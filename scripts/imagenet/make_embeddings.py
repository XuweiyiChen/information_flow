import os
import numpy as np
import torch

from experiments.utils.model_definitions.vision_automodel_wrapper import VisionLayerwiseAutoModelWrapper, VisionModelSpecifications
from experiments.utils.dataloaders.vision_dataloader import prepare_datasets, prepare_dataloader, validation_imagenet_transform
from experiments.utils.dataloaders.convert_to_embeddings import convert_image_dataset_to_embeddings
from experiments.utils.misc.optimal_batch_size import find_optimal_batch_size
from experiments.utils.dataloaders.convert_to_embeddings import convert_image_dataset_to_embeddings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers.utils import logging
logging.set_verbosity_error()


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


image_transform = validation_imagenet_transform()
train_dataset = prepare_datasets(
    dataset="imagenet",
    transform=image_transform,
    train_data_path="/home/AD/ofsk222/Research/exploration/information_plane/experiments/datasets/imagenet/ILSVRC/Data/CLS-LOC/train",
    number_of_samples=-1
)
val_dataset = prepare_datasets(
    dataset="imagenet",
    transform=image_transform,
    train_data_path="/home/AD/ofsk222/Research/exploration/information_plane/experiments/datasets/imagenet/ILSVRC/Data/CLS-LOC/val_sorted",
    number_of_samples=-1
)

models_to_try = [
    # # Base models
    # VisionModelSpecifications(model_family="dinov2", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="mae", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="clip", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="vit", model_size="base", revision="main"),

    # Small models
    # VisionModelSpecifications(model_family="beit", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="dinov2-register", model_size="small", revision="main"),
    
    # VisionModelSpecifications(model_family="i-jepa", model_size="imagenet1k", revision="main"),
    # VisionModelSpecifications(model_family="i-jepa", model_size="imagenet21k", revision="main"),
    # # Large/huge models
    # VisionModelSpecifications(model_family="dinov2", model_size="large", revision="main"),
    # VisionModelSpecifications(model_family="mae", model_size="large", revision="main"),
    # VisionModelSpecifications(model_family="clip", model_size="large", revision="main"),

    # # VisionModelSpecifications(model_family="dinov2-register", model_size="base", revision="main"),
    # # VisionModelSpecifications(model_family="dinov2-register", model_size="large", revision="main"),
    # # VisionModelSpecifications(model_family="dinov2-register", model_size="giant", revision="main"),

    # VisionModelSpecifications(model_family="mae", model_size="huge", revision="main"),
    # VisionModelSpecifications(model_family="vit", model_size="large", revision="main"),
    # VisionModelSpecifications(model_family="vit", model_size="huge", revision="main"),
    # VisionModelSpecifications(model_family="dinov2", model_size="giant", revision="main"),

    VisionModelSpecifications(model_family="aim", model_size="large", revision="main"),
    # VisionModelSpecifications(model_family="aim", model_size="huge", revision="main"),
    # VisionModelSpecifications(model_family="aim", model_size="1B", revision="main"),
    # VisionModelSpecifications(model_family="aim", model_size="3B", revision="main"),
]   

for model_spec in models_to_try:
    try:

        save_path = f"embeddings/{model_spec.model_family}/{model_spec.model_size}/imagenet"
        backbone = VisionLayerwiseAutoModelWrapper(model_specs=model_spec)

        optimal_batch_size = find_optimal_batch_size(backbone, 256, device=backbone._get_first_layer_device())
        print(model_spec)
        print(f"Optimal batch size: {optimal_batch_size}")

        train_save_path = f"{save_path}/train.pt"
        if not os.path.exists(train_save_path):
            train_dataloader = prepare_dataloader(train_dataset, batch_size=optimal_batch_size, num_workers=64, shuffle=True)
            convert_image_dataset_to_embeddings(train_dataloader, backbone, train_save_path)

        val_save_path = f"{save_path}/val.pt"
        if not os.path.exists(val_save_path):
            val_dataloader = prepare_dataloader(val_dataset, batch_size=optimal_batch_size, num_workers=64, shuffle=False)
            convert_image_dataset_to_embeddings(val_dataloader, backbone, val_save_path)

        del backbone
        torch.cuda.empty_cache()
    except Exception as e:
        #raise e
        print(f"Error with {model_spec}")
        print(e)
        raise e