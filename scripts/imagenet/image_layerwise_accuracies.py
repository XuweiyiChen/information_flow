import torch
import lightning as pl
import argparse
import omegaconf
import pickle
import os
from experiments.utils.model_definitions.probe.LinearProbe import LinearModel

# add args for model family, size, and layer
parser = argparse.ArgumentParser()
parser.add_argument("--model_family", type=str, default="dinov2")
parser.add_argument("--model_size", type=str, default="base")
parser.add_argument("--layer", type=int, default=-1)
args = parser.parse_args()


# load data
save_path = f"embeddings/{args.model_family}/{args.model_size}/imagenet"
loaded_train_dataset = torch.load(f"{save_path}/train.pt")
loaded_val_dataset = torch.load(f"{save_path}/val.pt")

# make datasets
train_dataloader = torch.utils.data.DataLoader(loaded_train_dataset, batch_size=4096, shuffle=True, num_workers=8)
val_dataloader = torch.utils.data.DataLoader(loaded_val_dataset, batch_size=4096, shuffle=False, num_workers=8)

# make model
cfg = omegaconf.OmegaConf.create({
    "data": {
        "num_classes": 1000,
    },
    "scheduler": {
        "lr_decay_steps": [60, 80],
    },
    "optimizer": {
        "weight_decay": 0.01,
        "name": "adam",
        "batch_size": 1024,
        "lr": 0.001,
    },
    "max_epochs": 10,
    "eval_layer": args.layer
})
probe = LinearModel(cfg=cfg)
trainer = pl.Trainer(max_epochs=cfg.max_epochs, callbacks=[], logger=False, devices=1,  precision='16-mixed')

# train model
trainer.fit(probe, train_dataloader, val_dataloader)

# save accuracies
accuracies = dict(probe.trainer.callback_metrics)
accuracies = {k: v.item() for k, v in accuracies.items()}

results_path = f"experiments/results/{args.model_family}/{args.model_size}/imagenet/layer_{args.layer}.pkl"
if not os.path.exists(results_path):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, "wb") as f:
    pickle.dump(accuracies, f)





