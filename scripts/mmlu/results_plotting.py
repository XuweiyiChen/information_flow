import torch
import pickle
import matplotlib.pyplot as plt
import os
import cmasher as cmr
import numpy as np

SIZES = ['70m', '160m', '410m', '1.4b', '2.8b']

colors = cmr.lavender(np.linspace(0.8, 0.2, len(SIZES)))

plt.figure(figsize=(10, 6))

for i, size in enumerate(SIZES):
    base_path = f"../../experiments/results/Pythia/{size}/main/mmlu"
    layers = []
    accuracies = []
    std_errs = []
    
    # Iterate through all layer files
    for filename in os.listdir(base_path):
        if filename.startswith("layer_"):
            # Extract layer number from filename
            layer_num = int(filename.split("_")[1])
            
            # Load results from pickle file
            path = os.path.join(base_path, filename, "tuned.pkl")
            with open(path, 'rb') as infile:
                results = pickle.load(infile)
                
            # Get accuracy from results
            accuracy = results['results']['mmlu']['acc,none']
            stderr = results['results']['mmlu']['acc_stderr,none']
            
            layers.append(layer_num)
            accuracies.append(accuracy)
            std_errs.append(stderr)
            
    # Sort by layer number
    sorted_pairs = sorted(zip(layers, accuracies, std_errs))
    layers, accuracies, std_errs = zip(*sorted_pairs)
    
    # Plot for this model size
    plt.errorbar(layers, accuracies, yerr=std_errs, marker='o', linestyle='-', 
                capsize=5, label=f'Pythia-{size}', color=colors[i])

plt.xlabel('Layer')
plt.ylabel('MMLU Accuracy')
plt.title('MMLU Accuracy vs Layer for Different Pythia Scales')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('../../mmlu_pythia_scales.png')
plt.close()