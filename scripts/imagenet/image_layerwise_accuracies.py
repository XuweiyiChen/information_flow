import torch
import lightning as pl
import argparse
import omegaconf
import pickle
import os
from experiments.utils.model_definitions.probe.LinearProbe import LinearModel

# add args for model family, size, and layer
parser = argparse.ArgumentParser()
parser.add_argument("--model_family", type=str, default="aim")
parser.add_argument("--model_size", type=str, default="1B")
parser.add_argument("--layer", type=int, default=-1)
args = parser.parse_args()

# load data
print('here')
save_path = f"embeddings/{args.model_family}/{args.model_size}/imagenet100"
loaded_train_dataset = torch.load(f"{save_path}/train.pt")
loaded_val_dataset = torch.load(f"{save_path}/val.pt")
print('here2')
# make datasets
train_dataloader = torch.utils.data.DataLoader(loaded_train_dataset, batch_size=4096, shuffle=True, num_workers=8)
val_dataloader = torch.utils.data.DataLoader(loaded_val_dataset, batch_size=4096, shuffle=False, num_workers=8)

for layer in reversed(range(0, 24)):
    args.layer = layer
    # if results already exist, skip
    results_path = f"experiments/results/{args.model_family}/{args.model_size}/imagenet100/layer_{args.layer}.pkl"
    if not os.path.exists(results_path):
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
    else:
        print(f"Results already exist for {args.model_family} {args.model_size} layer {args.layer}")
        continue


    probe = LinearModel(cfg=cfg)
    trainer = pl.Trainer(max_epochs=cfg.max_epochs, callbacks=[], logger=False, devices=1,  precision='16-mixed')

    # train model
    trainer.fit(probe, train_dataloader, val_dataloader)

    # save accuracies
    accuracies = dict(probe.trainer.callback_metrics)
    accuracies = {k: v.item() for k, v in accuracies.items()}

    with open(results_path, "wb") as f:
        pickle.dump(accuracies, f)





