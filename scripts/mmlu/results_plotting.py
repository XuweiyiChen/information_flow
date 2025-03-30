import torch
import pickle
import matplotlib.pyplot as plt
import os
import cmasher as cmr
import numpy as np

SIZES = ['410m']

colors = cmr.lavender(np.linspace(0.8, 0.2, len(SIZES)))
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 6))

for i, size in enumerate(SIZES):
    base_path = f"/home/AD/ofsk222/Research/exploration/information_plane/experiments/results/Pythia/{size}/main/toxigen"
    layers = []
    logit_accuracies = []
    logit_std_errs = []
    tuned_accuracies = []
    tuned_std_errs = []
    
    # Iterate through all layer files
    for filename in os.listdir(base_path):
        if filename.startswith("layer_"):
            # Extract layer number from filename
            layer_num = int(filename.split("_")[1])
            
            # Load results from logit pickle file
            logit_path = os.path.join(base_path, filename, "logit.pkl")
            if os.path.exists(logit_path):
                with open(logit_path, 'rb') as infile:
                    results = pickle.load(infile)
                    task_name = 'mmlu' if 'mmlu' in base_path else 'toxigen'
                    accuracy = results['results'][task_name]['acc,none']
                    stderr = results['results'][task_name]['acc_stderr,none']
                    layers.append(layer_num)
                    logit_accuracies.append(accuracy)
                    logit_std_errs.append(stderr)

            # Load results from tuned pickle file  
            tuned_path = os.path.join(base_path, filename, "tuned.pkl")
            if os.path.exists(tuned_path):
                with open(tuned_path, 'rb') as infile:
                    results = pickle.load(infile)
                    task_name = 'mmlu' if 'mmlu' in base_path else 'toxigen'
                    accuracy = results['results'][task_name]['acc,none']
                    stderr = results['results'][task_name]['acc_stderr,none']
                    tuned_accuracies.append(accuracy)
                    tuned_std_errs.append(stderr)
            
    # Sort by layer number
    sorted_pairs = sorted(zip(layers, logit_accuracies, logit_std_errs, tuned_accuracies, tuned_std_errs))
    layers, logit_accuracies, logit_std_errs, tuned_accuracies, tuned_std_errs = zip(*sorted_pairs)
    
    # Plot for this model size - both logit and tuned
    if False:
        plt.errorbar(layers, logit_accuracies, yerr=logit_std_errs, marker='o', linestyle='-', 
                    capsize=5, label=f'Logit Lens', color='red')
        plt.errorbar(layers, tuned_accuracies, yerr=tuned_std_errs, marker='s', linestyle='-',
                    capsize=5, label=f'Tuned Lens', color='blue')
    else:
        plt.plot(layers, logit_accuracies, marker='o', linestyle='-', 
                label=f'Logit Lens', color='red')
        plt.plot(layers, tuned_accuracies, marker='s', linestyle='-',
                 label=f'Tuned Lens', color='blue')

plt.xlabel('Layer')
plt.ylabel('Toxigen Accuracy')
plt.title('Layerwise Toxigen Accuracy on Pythia-410M')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('toxigen_410m.pdf')    
plt.close()