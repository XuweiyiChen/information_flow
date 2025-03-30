import torch
import os
import argparse
import pickle
from lm_eval import evaluator
from experiments.utils.model_definitions.mmlu.mmlu_harness_wrapper import PythiaLens

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Pythia', choices=['Pythia', 'Llama3'])
    parser.add_argument('--model_size', type=str, default='70m', choices=PythiaLens.VALID_SIZES)
    parser.add_argument('--evaluation_layer', type=int, default=-1, help='Layer to use for evaluation. -1 for the final layer. This is 0-indexed.')
    parser.add_argument('--base_results_path', type=str, default='experiments/results')
    parser.add_argument('--task', type=str, default='mmlu', choices=['mmlu', 'blimp', 'toxigen', 'crows_pairs_english'])
    parser.add_argument('--lens-type', type=str, default='tuned', choices=['logit', 'tuned'])
    return parser.parse_args()

def get_results_path(base_path, model_name, size, task, layer, lens_type):
    return f"{base_path}/{model_name}/{size}/main/{task}/layer_{layer}/{lens_type}.pkl"

def save_results(results, base_path, model_name, size, task, layer, lens_type):
    save_path = get_results_path(base_path, model_name, size, task, layer, lens_type)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

def evaluate():
    args = parse_args()
    
    model = PythiaLens(model_size=args.model_size, 
                       evaluation_layer=args.evaluation_layer,
                       lens_type=args.lens_type,
                       model_name=args.model_name)
    layer_name = args.evaluation_layer if args.evaluation_layer != -1 else model.config.num_hidden_layers
    
    # Check if results already exist
    results_path = get_results_path(args.base_results_path, args.model_name, args.model_size, args.task, layer_name, args.lens_type)
    if os.path.exists(results_path):
        print(f"Results already exist at {results_path}. Skipping evaluation.")
        return


    results = evaluator.simple_evaluate(
        model=model,
        tasks=args.task,
        verbosity="WARNING",
        batch_size='auto'
    )

    save_results(results, args.base_results_path, args.model_name, args.model_size, args.task, layer_name, args.lens_type)

if __name__ == "__main__":
    evaluate()
