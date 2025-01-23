import torch
import tqdm
import numpy as np
import itertools
import matplotlib.pyplot as plt
from experiments.utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications
from experiments.utils.model_definitions.vision_automodel_wrapper import VisionLayerwiseAutoModelWrapper, VisionModelSpecifications
from experiments.utils.dataloaders.vision_dataloader import prepare_datasets, prepare_dataloader, validation_imagenet_transform, simclr_imagenet_transform
from experiments.utils.misc.results_saving import load_results, check_if_results_exist
from experiments.utils.metrics.metric_calling import EvaluationMetricSpecifications, calculate_and_save_layerwise_metrics
from experiments.utils.misc.optimal_batch_size import find_optimal_batch_size

# models_to_try = [
#     VisionModelSpecifications(model_family="dinov2", model_size="small", revision="main"),
#     VisionModelSpecifications(model_family="dinov2", model_size="base", revision="main"),
#     VisionModelSpecifications(model_family="dinov2", model_size="large", revision="main"),
#     VisionModelSpecifications(model_family="dinov2", model_size="giant", revision="main"),
#     # VisionModelSpecifications(model_family="dinov2-register", model_size="small", revision="main"),
#     # VisionModelSpecifications(model_family="dinov2-register", model_size="base", revision="main"),
#     # VisionModelSpecifications(model_family="dinov2-register", model_size="large", revision="main"),
#     # VisionModelSpecifications(model_family="dinov2-register", model_size="giant", revision="main"),
#     VisionModelSpecifications(model_family="mae", model_size="base", revision="main"),
#     VisionModelSpecifications(model_family="mae", model_size="large", revision="main"),
#     VisionModelSpecifications(model_family="mae", model_size="huge", revision="main"),
#     VisionModelSpecifications(model_family="clip", model_size="base", revision="main"),
#     VisionModelSpecifications(model_family="clip", model_size="large", revision="main"),
#     VisionModelSpecifications(model_family="vit", model_size="base", revision="main"),
#     VisionModelSpecifications(model_family="vit", model_size="large", revision="main"),
#     VisionModelSpecifications(model_family="vit", model_size="huge", revision="main"),
#     VisionModelSpecifications(model_family="i-jepa", model_size="imagenet1k", revision="main"),
#     VisionModelSpecifications(model_family="i-jepa", model_size="imagenet21k", revision="main"),
# ]   

models_to_try = [
    TextModelSpecifications(
        model_family="Pythia",
        model_size="410m",
        revision="main"
    )
]

metrics_to_try = [
    EvaluationMetricSpecifications(evaluation_metric="sentence-entropy", alpha=2),
]

model_to_results = {}
for model_specs, evaluation_metric_specs in itertools.product(models_to_try, metrics_to_try):
    print(model_specs, evaluation_metric_specs)
    key = f"{model_specs.model_family}-{model_specs.model_size}"
    dataloader_kwargs = {"dataset_name": "imagenet"}

    if check_if_results_exist(model_specs, evaluation_metric_specs, dataloader_kwargs):
        model_to_results[key] = load_results(model_specs, evaluation_metric_specs, dataloader_kwargs)
        continue
    
    model = VisionLayerwiseAutoModelWrapper(model_specs, device_map="auto")

    if evaluation_metric_specs.evaluation_metric in ['lidar', 'infonce']:
        num_crops = 16 if evaluation_metric_specs.evaluation_metric == 'lidar' else 2
        image_transform = simclr_imagenet_transform(
            crop_size = (model.image_processor.crop_size['height'], model.image_processor.crop_size['width']),
            mean = model.image_processor.image_mean,
            std = model.image_processor.image_std,
            num_crops = num_crops
        )
    else:
        image_transform = validation_imagenet_transform()

    validation_imagenet_dataset = prepare_datasets(
        dataset="imagenet", 
        transform=image_transform,
        train_data_path="/home/AD/ofsk222/Research/exploration/information_plane/experiments/datasets",
        number_of_samples=1000
    )

    validation_dataloader = prepare_dataloader(validation_imagenet_dataset, batch_size=32, num_workers=64, shuffle=False)

    # # # save iamge of sample batch
    # # for batch in validation_dataloader:
    # #     single_view_single_image = [batch[i][1][2] for i in range(len(batch))]
    # #     break
    # # single_view_single_image = torch.stack(single_view_single_image)
    # # single_view_single_image = single_view_single_image.cpu().numpy().reshape(3, 224*16, 224)
    # single_view_single_image = np.transpose(single_view_single_image, (1, 2, 0))
    # # Normalize to 0-1 range
    # single_view_single_image = (single_view_single_image - single_view_single_image.min()) / (single_view_single_image.max() - single_view_single_image.min())
    # plt.imsave("sample_batch.png", single_view_single_image)


    results = calculate_and_save_layerwise_metrics(model, validation_dataloader, model_specs, evaluation_metric_specs, dataloader_kwargs)

    del model
    torch.cuda.empty_cache()