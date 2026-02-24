#!/usr/bin/env python3
"""
Plot MTEB main_score across layers for gated attention models.

Usage:
    python scripts/plot_mteb_layer_accuracy.py \
        --results_base experiments/results \
        --models gated_attention_baseline gated_attention_elementwise gated_attention_headwise \
        --size 1B \
        --revision main \
        --output plots/mteb_layer_accuracy.png
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np


def load_results(results_base, model_family, size, revision):
    """Load main_score for each task at each layer."""
    mteb_dir = Path(results_base) / model_family / size / revision / "mteb"
    if not mteb_dir.exists():
        print(f"Warning: {mteb_dir} does not exist, skipping {model_family}")
        return {}, []

    layer_scores = {}
    task_names = set()

    for layer_dir in sorted(mteb_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        try:
            layer_idx = int(layer_dir.name.split("_")[1])
        except ValueError:
            continue

        scores = {}
        for task_file in layer_dir.glob("*.json"):
            if task_file.name == "model_meta.json":
                continue
            task_name = task_file.stem
            with open(task_file) as f:
                data = json.load(f)
            try:
                main_score = data["scores"]["test"][0]["main_score"]
                scores[task_name] = main_score
                task_names.add(task_name)
            except (KeyError, IndexError):
                continue

        if scores:
            layer_scores[layer_idx] = scores

    return layer_scores, sorted(task_names)


def compute_category_averages(layer_scores, task_names):
    """Group tasks by MTEB type and compute per-layer averages."""
    classification_tasks = [
        "AmazonCounterfactualClassification", "AmazonReviewsClassification",
        "Banking77Classification", "EmotionClassification",
        "MassiveIntentClassification", "MassiveScenarioClassification",
        "MTOPDomainClassification", "MTOPIntentClassification",
        "ToxicConversationsClassification", "TweetSentimentExtractionClassification",
    ]
    clustering_tasks = [
        "ArxivClusteringS2S", "BiorxivClusteringS2S", "MedrxivClusteringS2S",
        "RedditClustering", "StackExchangeClustering", "TwentyNewsgroupsClustering",
    ]
    sts_tasks = [
        "BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16",
        "STS17", "STSBenchmark",
    ]
    pair_tasks = [
        "AskUbuntuDupQuestions", "SprintDuplicateQuestions",
        "StackOverflowDupQuestions", "TwitterSemEval2015", "TwitterURLCorpus",
    ]
    reranking_tasks = ["MindSmallReranking", "SciDocsRR"]

    categories = {
        "Classification": classification_tasks,
        "Clustering": clustering_tasks,
        "STS": sts_tasks,
        "PairClassification": pair_tasks,
        "Reranking": reranking_tasks,
    }

    layers = sorted(layer_scores.keys())
    category_avgs = {}

    for cat_name, cat_tasks in categories.items():
        avgs = []
        for layer in layers:
            scores = layer_scores.get(layer, {})
            vals = [scores[t] for t in cat_tasks if t in scores]
            avgs.append(np.mean(vals) if vals else np.nan)
        category_avgs[cat_name] = avgs

    overall = []
    for layer in layers:
        scores = layer_scores.get(layer, {})
        vals = list(scores.values())
        overall.append(np.mean(vals) if vals else np.nan)
    category_avgs["Overall"] = overall

    return layers, category_avgs


def main():
    parser = argparse.ArgumentParser(description="Plot MTEB scores across layers")
    parser.add_argument("--results_base", type=str, default="experiments/results")
    parser.add_argument("--models", nargs="+", default=[
        "gated_attention_baseline",
        "gated_attention_elementwise",
        "gated_attention_headwise",
    ])
    parser.add_argument("--size", type=str, default="1B")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--output", type=str, default="plots/mteb_layer_accuracy.png")
    args = parser.parse_args()

    model_colors = {
        "gated_attention_baseline": "#1f77b4",
        "gated_attention_elementwise": "#ff7f0e",
        "gated_attention_headwise": "#2ca02c",
    }
    model_labels = {
        "gated_attention_baseline": "Baseline",
        "gated_attention_elementwise": "Elementwise Gate",
        "gated_attention_headwise": "Headwise Gate",
    }

    all_data = {}
    for model in args.models:
        layer_scores, task_names = load_results(args.results_base, model, args.size, args.revision)
        if layer_scores:
            layers, cat_avgs = compute_category_averages(layer_scores, task_names)
            all_data[model] = (layers, cat_avgs)
            print(f"Loaded {model}: {len(layers)} layers, {len(task_names)} tasks")
        else:
            print(f"No results found for {model}")

    if not all_data:
        print("No data to plot.")
        return

    categories = ["Overall", "Classification", "Clustering", "STS", "PairClassification", "Reranking"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    for ax, cat in zip(axes, categories):
        for model in args.models:
            if model not in all_data:
                continue
            layers, cat_avgs = all_data[model]
            if cat in cat_avgs:
                ax.plot(layers, cat_avgs[cat],
                        color=model_colors.get(model, None),
                        label=model_labels.get(model, model),
                        linewidth=2, marker="o", markersize=3)

        ax.set_title(cat, fontsize=14, fontweight="bold")
        ax.set_ylabel("Main Score")
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"MTEB Scores by Layer — Gated Attention Models ({args.size})",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
