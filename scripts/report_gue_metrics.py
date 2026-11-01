"""Report saved test predictions without retraining or changing source results.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_gue as gue


def scalar_metrics(metrics):
    """Flatten performance metrics, retaining original class labels in names."""
    values = {key: value for key, value in metrics.items()
              if value is None or isinstance(value, (int, float))}
    for row in metrics.get("per_class", []):
        for key in ("precision", "recall", "f1", "auc_ovr"):
            values[f"class_{row['label']}/{key}"] = row[key]
    return values


def write_all_metric_summaries(folder, runs, dataset_id, seeds):
    """Summarize every available dev/test score; undefined scores remain empty."""
    rows = []
    info = runs[0].get("source_config", {}).get("dataset", {})
    counts = info.get("audit", {}).get("rows", {})
    metadata = {"dataset_name": Path(dataset_id).name, "folder": info.get("folder"),
                "num_classes": info.get("num_labels"),
                **{f"{split}_rows": counts.get(split) for split in ("train", "dev", "test")},
                "selection_metric": runs[0].get("selection_metric"),
                "selection_mode": runs[0].get("selection_mode")}
    for split in ("best_dev", "test"):
        flattened = [scalar_metrics(run.get(split, {})) for run in runs]
        for metric in sorted(set().union(*(values.keys() for values in flattened))):
            values = [scores.get(metric) for scores in flattened]
            valid = [value for value in values if value is not None]
            if any(not math.isfinite(value) for value in valid):
                raise ValueError(f"Nonfinite {split} {metric} for {dataset_id}.")
            mean = float(np.mean(valid)) if valid else None
            sd = float(np.std(valid, ddof=1)) if len(valid) > 1 else None
            rows.append({"dataset": dataset_id, **metadata, "split": split, "metric": metric,
                         **{f"seed_{seed}": value for seed, value in zip(seeds, values)},
                         "n": len(valid), "requested_seeds": len(seeds), "mean": mean, "sd": sd,
                         "mean_sd": f"{mean:.6f} ± {sd:.6f}" if sd is not None else ""})
    gue.write_json(Path(folder) / "metrics_summary.json", rows)
    with (Path(folder) / "metrics_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", *metadata, "split", "metric",
                                                   *[f"seed_{seed}" for seed in seeds],
                                                   "n", "requested_seeds", "mean", "sd", "mean_sd"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def report_saved_metrics(input_dir, output_dir, data_root, seeds):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("Choose at least two distinct seeds in [0, 2**32).")
    source = Path(input_dir).resolve()
    destination = Path(output_dir).resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Use a separate output directory to preserve source results.")
    grouped = defaultdict(dict)
    for result_path in sorted(source.rglob("test_results.json")):
        run = json.loads(result_path.read_text())
        seed = run["seed"]
        if seed not in seeds:
            continue
        info = run["config"]["dataset"]
        dataset_id = gue.dataset_identifier(info["folder"], data_root)
        if seed in grouped[dataset_id]:
            raise ValueError(f"Multiple results for {dataset_id} seed {seed}.")
        classes = info.get("class_labels", list(range(info["num_labels"])))
        mapping = {label: index for index, label in enumerate(classes)}
        with (result_path.parent / "test_predictions.csv").open(newline="") as handle:
            predictions = list(csv.DictReader(handle))
        labels = [mapping[int(row["label"])] for row in predictions]
        if len(classes) == 2:
            probabilities = [float(row["probability"]) for row in predictions]
            predicted = [classes[int(probability >= 0.5)] for probability in probabilities]
        else:
            probabilities = [[float(row[f"probability_{label}"]) for label in classes] for row in predictions]
            predicted = [classes[int(np.argmax(row))] for row in probabilities]
        if predicted != [int(row["prediction"]) for row in predictions]:
            raise ValueError(f"Saved predictions disagree with probabilities: {result_path}")
        if len(labels) != info["audit"]["rows"]["test"]:
            raise ValueError(f"Saved prediction count disagrees with dataset metadata: {result_path}")
        recipe = run["config"]["recipe"]
        strategy = recipe.get("multiclass_auc_strategy", recipe.get("ss_auc_strategy"))
        metrics = gue.calculate_metric_with_sklearn(probabilities, labels, class_labels=classes)
        # Keep the exact model loss, rather than recomputing it from rounded probabilities.
        metrics["loss"] = run["test"]["loss"]
        for key in ("accuracy", "mcc", "auc"):
            reported_key = f"auc_{strategy}" if key == "auc" and len(classes) > 2 else key
            if key in run["test"] and not np.isclose(metrics[reported_key], run["test"][key], rtol=1e-7, atol=1e-9):
                raise ValueError(f"Recomputed {key} disagrees with source results: {result_path}")
        best_dev = dict(run.get("best_dev", {}))
        for key in ("specificity", "average_precision", "average_precision_macro", "average_precision_weighted",
                    "confusion_matrix", "support"):
            best_dev.pop(key, None)
        if "per_class" in best_dev:
            best_dev["per_class"] = [{key: row[key] for key in ("label", "precision", "recall", "f1", "auc_ovr")}
                                     for row in best_dev["per_class"]]
        report = {"dataset": dataset_id, "dataset_path": info["folder"], "seed": seed,
                  "selection_metric": run["selection_metric"], "selection_mode": run.get("selection_mode"),
                  "best_epoch": run.get("best_epoch"), "best_dev": best_dev, "test": metrics,
                  "source_result": str(result_path), "source_config": run["config"],
                  "report_scope": "Saved test predictions from the recorded original experiment; no retraining."}
        grouped[dataset_id][seed] = report
    if not grouped:
        raise FileNotFoundError(f"No saved runs in {source}.")
    # Validate every seed and recipe before writing reports.
    for dataset_id, runs in grouped.items():
        if set(runs) != set(seeds):
            raise ValueError(f"Missing requested seeds for {dataset_id}.")
        common = [{k: v for k, v in run["source_config"].items() if k != "seed"} for run in runs.values()]
        if any(config != common[0] for config in common[1:]):
            raise ValueError(f"Different source recipes or splits for {dataset_id}.")
    combined = []
    for dataset_id, runs in sorted(grouped.items()):
        spec = gue.DatasetSpec(Path(runs[seeds[0]]["dataset_path"]), dataset_id)
        folder = gue.dataset_output_dir(destination, spec)
        for seed in seeds:
            seed_dir = folder / f"seed_{seed}"
            gue.write_json(seed_dir / "metrics.json", runs[seed])
            gue.write_metric_details(seed_dir, runs[seed]["test"])
        combined.extend(write_all_metric_summaries(folder, [runs[seed] for seed in seeds], dataset_id, seeds))
    gue.write_json(destination / "metrics_summary.json", combined)
    with (destination / "metrics_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    return {"datasets": sorted(grouped), "seeds": seeds, "output_dir": str(destination)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/evo2_gue")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="GUE")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run reporting on a GPU.")
    print(json.dumps(report_saved_metrics(args.input_dir, args.output_dir, args.data_root, args.seeds), indent=2))
