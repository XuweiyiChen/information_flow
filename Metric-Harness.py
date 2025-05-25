import torch
import tqdm
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from experiments.utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications
from experiments.utils.model_definitions.vision_automodel_wrapper import VisionLayerwiseAutoModelWrapper, VisionModelSpecifications
from experiments.utils.dataloaders.vision_dataloader import prepare_datasets, prepare_dataloader, validation_imagenet_transform, simclr_imagenet_transform, ImageDatasetFromDirectory
from experiments.utils.misc.results_saving import construct_file_path, load_results, check_if_results_exist
from experiments.utils.metrics.metric_calling import EvaluationMetricSpecifications, calculate_and_save_layerwise_metrics
from experiments.utils.misc.optimal_batch_size import find_optimal_batch_size

models_to_try = [
    # VisionModelSpecifications(model_family="beit", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="beit", model_size="large", revision="main"),
    # #VisionModelSpecifications(model_family="dinov2", model_size="small", revision="main"),
    # VisionModelSpecifications(model_family="dinov2", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="dinov2", model_size="large", revision="main"),
    # #VisionModelSpecifications(model_family="dinov2", model_size="giant", revision="main"),
    # # VisionModelSpecifications(model_family="dinov2-register", model_size="small", revision="main"),
    # # VisionModelSpecifications(model_family="dinov2-register", model_size="base", revision="main"),
    # # VisionModelSpecifications(model_family="dinov2-register", model_size="large", revision="main"),
    # # VisionModelSpecifications(model_family="dinov2-register", model_size="giant", revision="main"),
    # VisionModelSpecifications(model_family="mae", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="mae", model_size="large", revision="main"),
    # #VisionModelSpecifications(model_family="mae", model_size="huge", revision="main"),
    # VisionModelSpecifications(model_family="clip", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="clip", model_size="large", revision="main"),
    # VisionModelSpecifications(model_family="vit", model_size="base", revision="main"),
    # VisionModelSpecifications(model_family="vit", model_size="large", revision="main"),
    # #VisionModelSpecifications(model_family="vit", model_size="huge", revision="main"),
    # VisionModelSpecifications(model_family="i-jepa", model_size="imagenet1k", revision="main"),
    # VisionModelSpecifications(model_family="i-jepa", model_size="imagenet21k", revision="main"),
    VisionModelSpecifications(model_family="clip", model_size="large", revision="main"),
    VisionModelSpecifications(model_family="dinov2", model_size="large", revision="main"),
    VisionModelSpecifications(model_family="mae", model_size="large", revision="main"),
    VisionModelSpecifications(model_family="vit", model_size="large", revision="main"),
    VisionModelSpecifications(model_family="aim", model_size="large", revision="main"),
    VisionModelSpecifications(model_family="aimv2", model_size="large", revision="main"),

    VisionModelSpecifications(model_family="beit", model_size="large", revision="main"),
    # VisionModelSpecifications(model_family="aim", model_size="huge", revision="main"),
    #VisionModelSpecifications(model_family="aim", model_size="1B", revision="main"),
    #VisionModelSpecifications(model_family="aim", model_size="3B", revision="main"),
    # VisionModelSpecifications(model_family="aim", model_size="7B", revision="main"),
]   

# models_to_try = [
#     TextModelSpecifications(
#         model_family="Pythia",
#         model_size="410m",
#         revision="main"
#     )
# ]

metrics_to_try = [
    EvaluationMetricSpecifications(evaluation_metric="prompt-entropy", alpha=1),
    # EvaluationMetricSpecifications(evaluation_metric="dataset-entropy", alpha=1),
    # EvaluationMetricSpecifications(evaluation_metric="infonce"),
    # EvaluationMetricSpecifications(evaluation_metric="dime"),
    EvaluationMetricSpecifications(evaluation_metric="lidar"),
]

model_to_results = {}
for model_specs, evaluation_metric_specs in itertools.product(models_to_try, metrics_to_try):
    key = f"{model_specs.model_family}-{model_specs.model_size}"
    dataloader_kwargs = {"dataset_name": "imagenet", "num_samples": 1000, "split": "val"}

    results_path = construct_file_path(
        model_specs, 
        evaluation_metric_specs, 
        dataloader_kwargs, 
        include_split=True
    )
    if os.path.exists(results_path):
        print(f"Results already exist for {model_specs.model_family} - {evaluation_metric_specs.evaluation_metric} - {dataloader_kwargs['split']}. Skipping...")
        continue

    print(model_specs, evaluation_metric_specs)

    model = VisionLayerwiseAutoModelWrapper(model_specs, device_map="auto")

    if evaluation_metric_specs.evaluation_metric in ['lidar', 'infonce', 'dime']:
        num_crops = 16 if evaluation_metric_specs.evaluation_metric == 'lidar' else 2
        image_transform = simclr_imagenet_transform(
            crop_size = (224, 224), #model.image_processor.crop_size['height'], model.image_processor.crop_size['width']),
            mean = model.image_processor.image_mean,
            std = model.image_processor.image_std,
            num_crops = num_crops
        )

        is_multiview = True
    else:
        image_transform = validation_imagenet_transform()
        is_multiview = False

    validation_imagenet_dataset = prepare_datasets(
        dataset="imagenet100", 
        transform=image_transform,
        train_data_path="/home/AD/ofsk222/Research/exploration/information_plane/experiments/datasets/imagenet100/val",
        number_of_samples=dataloader_kwargs["num_samples"]
    )

    # validation_imagenet_dataset = ImageDatasetFromDirectory(
    #     directory="/home/mila/a/arefinmr/scratch/LLM/information_flow/VisionCausal/data",
    #     transform=val_transforms(),
    #     n=dataloader_kwargs["num_samples"]
    # )

    optimal_batch_size = 32 #find_optimal_batch_size(model, len(validation_imagenet_dataset), device=model.device)
    validation_dataloader = prepare_dataloader(validation_imagenet_dataset, 
                                               batch_size=optimal_batch_size, 
                                               num_workers=4, 
                                               shuffle=False, 
                                               drop_last=False,
                                               is_multiview=is_multiview)

    # save image of sample batch
    for batch in validation_dataloader:
        if 'entropy' in evaluation_metric_specs.evaluation_metric:
            idx, images, targets = batch
            images_to_save = [images[i] for i in range(16)]
            grid_size = (4, 4)
        elif 'lidar' in evaluation_metric_specs.evaluation_metric:
            images_to_save = []
            for view in batch:
                # 16 views
                idx, images, targets = view
                images_to_save.append(images[0])
            grid_size = (4, 4)
        elif 'infonce' in evaluation_metric_specs.evaluation_metric or 'dime' in evaluation_metric_specs.evaluation_metric:
            images_to_save = []
            for view in batch:
                # 2 views
                idx, images, targets = view
                images_to_save.extend([images[i] for i in range(8)])
            grid_size = (4, 4)
            images_to_save[4:8], images_to_save[8:12] = images_to_save[8:12], images_to_save[4:8] # to make things line up
        else:
            raise ValueError(f"Unknown evaluation metric: {evaluation_metric_specs.evaluation_metric}")
        break

    images_to_save = torch.stack(images_to_save)
    images_to_save = images_to_save.cpu().numpy()
    
    # Create a grid of images
    h, w = 224, 224 
    grid = np.zeros((h * grid_size[0], w * grid_size[1], 3))
    
    for idx, img in enumerate(images_to_save):
        i = idx // grid_size[1]
        j = idx % grid_size[1]
       
        img = np.transpose(img, (1, 2, 0))
        grid[i*h:(i+1)*h, j*w:(j+1)*w] = img

    # Normalize to 0-1 range
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    plt.imsave(f"{evaluation_metric_specs.evaluation_metric}-sample_batch.png", grid)


    results = calculate_and_save_layerwise_metrics(model, validation_dataloader, model_specs, evaluation_metric_specs, dataloader_kwargs)

    del model
    torch.cuda.empty_cache()