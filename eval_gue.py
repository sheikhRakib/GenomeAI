import csv
import json
import logging
import gc
import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Dict, Sequence, List

import torch
import transformers
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from evo2 import Evo2
from vortex.model.tokenizer import CharLevelTokenizer

from peft import (
    LoraConfig,
    inject_adapter_in_model,
)


@dataclass
class ModelArguments:
    local_checkpoint: Optional[str] = field(default=None, metadata={"help": "Optional local Evo2 checkpoint file."})
    use_lora: bool = field(default=True, metadata={"help": "Use Wqkv LoRA; false trains only the classifier."})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None, metadata={"help": "One dataset folder containing sequence,label CSVs."})
    data_root: str = field(default="GUE")
    datasets: List[str] = field(default_factory=list, metadata={"help": "Dataset folders relative to data_root, or absolute paths."})


@dataclass(frozen=True)
class DatasetSpec:
    folder: Path
    identifier: str

    @property
    def name(self):
        return self.folder.name


@dataclass
class TrainingArguments:
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=4)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    num_train_epochs: int = field(default=5)
    bf16: bool = field(default=True)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=5e-3)
    learning_rate: float = field(default=3e-5)
    max_grad_norm: float = field(default=1.0)
    num_workers: int = field(default=0)
    logging_steps: int = field(default=100)
    output_dir: str = field(default="results/evo2_gue_datasets")
    seed: int = field(default=42, metadata={"help": "Random seed for this training run."})


def load_evo2_backbone(model_args: ModelArguments):
    """Load Evo2 7B and its native character tokenizer.

    Return the underlying PyTorch model: Evo2's public forward wrapper disables
    gradients, so it cannot be used for the later LoRA training step.
    """
    evo_model = Evo2("evo2_7b", local_path=model_args.local_checkpoint)
    prepare_backbone_for_training(evo_model.model)
    return evo_model.model, evo_model.tokenizer


def prepare_backbone_for_training(backbone: torch.nn.Module):
    """Make tensors produced by Vortex's inference-mode loader autograd-safe."""
    replacements = {}
    count = 0
    with torch.inference_mode(False), torch.no_grad():
        for module in backbone.modules():
            for name, parameter in list(module.named_parameters(recurse=False)):
                if torch.is_inference(parameter) or torch.is_inference(parameter.data):
                    if id(parameter) not in replacements:
                        replacements[id(parameter)] = torch.nn.Parameter(
                            parameter.detach().clone(), requires_grad=parameter.requires_grad,
                        )
                    module.register_parameter(name, replacements[id(parameter)])
                    count += 1
            for name, buffer in list(module.named_buffers(recurse=False)):
                if torch.is_inference(buffer):
                    if id(buffer) not in replacements:
                        replacements[id(buffer)] = buffer.clone()
                    persistent = name not in module._non_persistent_buffers_set
                    module.register_buffer(name, replacements[id(buffer)], persistent=persistent)
                    count += 1
    logging.info("Converted %s inference tensors for training", count)
    return backbone


class Evo2ForSequenceClassification(torch.nn.Module):
    """Classify DNA using masked mean pooling of Evo2's final hidden states."""

    def __init__(self, backbone: torch.nn.Module, pad_token_id: int, num_labels: int = 2):
        super().__init__()
        self.backbone = backbone
        self.pad_token_id = pad_token_id
        if num_labels < 2:
            raise ValueError("Classification requires at least two classes.")
        self.num_labels = num_labels

        # Native Evo2 applies final normalization before unembed. Replacing only
        # that vocabulary projection exposes hidden states with gradients intact.
        self.backbone.unembed = torch.nn.Identity()
        device = next(self.backbone.parameters()).device
        self.classifier = torch.nn.Linear(
            self.backbone.config.hidden_size, self.num_labels,
            device=device, dtype=torch.float32,
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence_length].")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        lengths = attention_mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Each sequence must contain at least one non-padding token.")

        # Call the native nn.Module rather than Evo2's no-grad public wrapper.
        hidden_states, _ = self.backbone(input_ids, padding_mask=attention_mask)
        mask = attention_mask.to(device=hidden_states.device)
        # Accumulate in fp32 and exclude padded states entirely from the mean.
        hidden_states = hidden_states.float().masked_fill(~mask.unsqueeze(-1), 0)
        pooled = hidden_states.sum(dim=1) / lengths.to(hidden_states.device).unsqueeze(-1)
        pooled = pooled.to(
            device=self.classifier.weight.device, dtype=self.classifier.weight.dtype,
        )
        logits = self.classifier(pooled)
        outputs = {"logits": logits}
        if labels is not None:
            outputs["loss"] = torch.nn.functional.cross_entropy(
                logits.float(), labels.to(device=logits.device, dtype=torch.long),
            )
        return outputs


def configure_wqkv_lora(model: Evo2ForSequenceClassification, args: ModelArguments):
    """Freeze Evo2 and inject adapters into its attention Wqkv projections only."""
    if args.lora_r <= 0 or args.lora_alpha <= 0 or not 0 <= args.lora_dropout < 1:
        raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0, 1).")
    if hasattr(model.backbone, "peft_config"):
        raise ValueError("LoRA has already been configured for this backbone.")

    model.backbone.requires_grad_(False)
    model.classifier.requires_grad_(True)
    model.lora_target_modules = []
    if args.use_lora:
        targets = [
            f"blocks.{index}.inner_mha_cls.Wqkv"
            for index in model.backbone.config.attn_layer_idxs
        ]
        if not targets:
            raise ValueError("No attention Wqkv projections found in the Evo2 config.")
        for name in targets:
            if not isinstance(model.backbone.get_submodule(name), torch.nn.Linear):
                raise TypeError(f"Unsupported Wqkv projection at {name}.")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            inference_mode=False,
        )
        # Inject into the native backbone to preserve its forward/padding API.
        # Vortex's dotdict reports every attribute as present (including a
        # nonexistent to_dict). Give PEFT a plain dict during configuration,
        # then restore the object Vortex uses for attribute access in forward.
        native_config = model.backbone.config
        model.backbone.config = dict(native_config)
        try:
            model.backbone = inject_adapter_in_model(config, model.backbone)
        finally:
            model.backbone.config = native_config
        # Keep optimizer weights in fp32 so small updates are not rounded away
        # by bf16 storage. Forward computation can still use bf16 autocast.
        for target in targets:
            module = model.backbone.get_submodule(target)
            module.lora_A["default"].to(dtype=torch.float32)
            module.lora_B["default"].to(dtype=torch.float32)
        model.lora_target_modules = targets

    allowed = {"classifier.weight", "classifier.bias"}
    for target in model.lora_target_modules:
        allowed.update({
            f"backbone.{target}.lora_A.default.weight",
            f"backbone.{target}.lora_B.default.weight",
        })
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if trainable != allowed:
        raise RuntimeError(f"Unexpected trainable parameters: {sorted(trainable ^ allowed)}")
    logging.warning(
        "Trainable parameters: %s across %s Wqkv adapters and the linear head",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        len(model.lora_target_modules),
    )
    return model


def read_classification_csv(path, num_labels=None):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sequence", "label"]:
            raise ValueError(f"{path}: expected sequence,label columns.")
        rows = []
        for line, row in enumerate(reader, start=2):
            sequence = row["sequence"]
            try:
                label = int(row["label"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{line}: label must be an integer.") from None
            if (not sequence or set(sequence) - set("ACGTN") or str(label) != row["label"]
                    or None in row or (num_labels is not None and label not in range(num_labels))):
                raise ValueError(f"{path}:{line}: invalid DNA sequence or class label.")
            rows.append({"sequence": sequence, "label": label})
    if not rows:
        raise ValueError(f"{path}: no data rows.")
    return rows


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: Optional[str],
                 tokenizer: CharLevelTokenizer,
                 model_max_length: int = 512,
                 rows=None,
                 num_labels: int = 2):

        super(SupervisedDataset, self).__init__()
        if model_max_length <= 0:
            raise ValueError("model_max_length must be positive.")

        self.rows = read_classification_csv(data_path, num_labels) if rows is None else rows
        if not self.rows:
            raise ValueError("Dataset must contain at least one row.")

        # Tokenize raw DNA; padding is handled by the collator.
        self.input_ids = [
            torch.tensor(tokenizer.tokenize(row["sequence"][:model_max_length]), dtype=torch.long)
            for row in self.rows
        ]
        self.labels = [row["label"] for row in self.rows]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: CharLevelTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_id
        )
        labels = torch.tensor(labels, dtype=torch.long)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_id),
        )

def calculate_metric_with_sklearn(probabilities, labels, class_labels=None):
    """Calculate all classification metrics, omitting undefined aggregate AUC."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0 or probabilities.ndim == 0 or probabilities.shape[0] != len(labels):
        raise ValueError("Expected one probability vector per label.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Class probabilities must be finite and in [0, 1].")
    if probabilities.ndim == 1:
        if not set(labels.tolist()) <= {0, 1}:
            raise ValueError("Binary evaluation expects encoded labels 0 and 1.")
        predictions = (probabilities >= 0.5).astype(np.int64)
    elif probabilities.ndim == 2 and probabilities.shape[1] >= 3:
        if not set(labels.tolist()) <= set(range(probabilities.shape[1])):
            raise ValueError("Multiclass labels are outside the encoded class range.")
        if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Class probabilities must sum to one.")
        predictions = probabilities.argmax(axis=1)
    else:
        raise ValueError("Use positive-class probabilities for binary data or full vectors for multiclass data.")
    num_classes = 2 if probabilities.ndim == 1 else probabilities.shape[1]
    original_labels = list(range(num_classes)) if class_labels is None else list(class_labels)
    if len(original_labels) != num_classes or len(set(original_labels)) != num_classes:
        raise ValueError("class_labels must identify every probability column exactly once.")
    encoded_classes = np.arange(num_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=encoded_classes, zero_division=0,
    )
    metrics = {"mcc": float(matthews_corrcoef(labels, predictions)),
               "accuracy": float(accuracy_score(labels, predictions))}
    for average in ("macro", "micro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            labels, predictions, labels=encoded_classes, average=average, zero_division=0,
        )
        metrics.update({f"precision_{average}": float(p), f"recall_{average}": float(r),
                        f"f1_{average}": float(f)})
    per_class = []
    for index, original_label in enumerate(original_labels):
        binary_labels = (labels == index).astype(np.int64)
        scores = ((probabilities if index == 1 else 1 - probabilities)
                  if num_classes == 2 else probabilities[:, index])
        defined = len(np.unique(binary_labels)) == 2
        per_class.append({"label": original_label, "precision": float(precision[index]),
                          "recall": float(recall[index]), "f1": float(f1[index]),
                          "auc_ovr": float(roc_auc_score(binary_labels, scores)) if defined else None})
    if num_classes == 2:
        positive = per_class[1]
        metrics.update({key: positive[key] for key in ("precision", "recall", "f1")})
    metrics["per_class"] = per_class
    if num_classes > 2 and set(labels.tolist()) == set(range(num_classes)):
        for strategy in ("ovr", "ovo"):
            metrics[f"auc_{strategy}"] = float(roc_auc_score(
                labels, probabilities, average="macro", multi_class=strategy,
            ))
    if num_classes == 2 and set(labels.tolist()) == {0, 1}:
        metrics["auc"] = per_class[1]["auc_ovr"]
    return metrics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dataset_identifier(folder, data_root):
    """Retain dataset subdirectories so repeated folder names do not collide."""
    folder = Path(folder).resolve()
    try:
        relative = folder.relative_to(Path(data_root).resolve())
    except ValueError:
        return folder.name
    return relative.as_posix() if relative.parts else folder.name


def resolve_datasets(args):
    if args.data_path and args.datasets:
        raise ValueError("Choose data_path for one dataset or datasets for multiple folders.")
    root = Path(args.data_root).resolve()
    if args.data_path:
        folders = [Path(args.data_path).resolve()]
    elif args.datasets:
        folders = [(Path(p) if Path(p).is_absolute() else root / p).resolve() for p in args.datasets]
    else:
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        folders = sorted({p.parent.resolve() for p in root.rglob("train.csv")})
    if not folders or len(set(folders)) != len(folders):
        raise ValueError("Choose distinct dataset folders containing train.csv.")
    datasets = [DatasetSpec(folder, dataset_identifier(folder, root)) for folder in folders]
    if len({d.identifier for d in datasets}) != len(datasets):
        raise ValueError("Dataset output names collide; choose a common data_root or separate output directories.")
    return datasets


def audit_splits(splits, max_length):
    sequences = {s: {r["sequence"][:max_length] for r in rows} for s, rows in splits.items()}
    labels = defaultdict(set)
    for rows in splits.values():
        for row in rows:
            labels[row["sequence"][:max_length]].add(row["label"])
    return {
        "rows": {s: len(rows) for s, rows in splits.items()},
        "label_counts": {s: {str(label): count for label, count in Counter(r["label"] for r in rows).items()}
                         for s, rows in splits.items()},
        "duplicate_sequence_label_rows": {
            s: len(rows) - len({(r["sequence"][:max_length], r["label"]) for r in rows})
            for s, rows in splits.items()},
        "sequences_with_conflicting_labels": sum(len(v) > 1 for v in labels.values()),
        "sequence_overlap": {f"{a}/{b}": len(sequences[a] & sequences[b])
                             for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))},
        "comparison": f"Exact input sequences after truncation to {max_length} characters",
    }


def prepare_dataset_data(dataset, max_length):
    folder = dataset.folder
    source_paths = [folder / f"{split}.csv" for split in ("train", "dev", "test")]
    missing = [path.name for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset files for {dataset.identifier}: {', '.join(missing)}")
    splits = {path.stem: read_classification_csv(path) for path in source_paths}
    class_labels = sorted({r["label"] for r in splits["train"]})
    num_labels = len(class_labels)
    if num_labels < 2:
        raise ValueError(f"{dataset.identifier}/train must contain at least two classes.")
    for split, rows in splits.items():
        labels = {r["label"] for r in rows}
        if not labels <= set(class_labels):
            raise ValueError(f"{dataset.identifier}/{split} contains labels absent from train.csv.")
    source_hashes = {}
    for path in source_paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        source_hashes[path.name] = digest.hexdigest()
    info = {"name": dataset.name, "identifier": dataset.identifier,
            "folder": str(folder), "source_sha256": source_hashes, "num_labels": num_labels,
            "class_labels": class_labels,
            "label_to_index": {str(label): index for index, label in enumerate(class_labels)},
            "positive_label": class_labels[1] if num_labels == 2 else None,
            "split_source": "provided_csvs",
            "audit": audit_splits(splits, max_length)}
    audit = info["audit"]
    if audit["sequences_with_conflicting_labels"] or any(audit["sequence_overlap"].values()):
        logging.warning("%s dataset has label conflicts or split overlap: %s", dataset.identifier, audit)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    splits = {split: [{"sequence": row["sequence"], "label": label_to_index[row["label"]]}
                      for row in rows] for split, rows in splits.items()}
    return splits, info


def save_trainable_checkpoint(model, path, epoch, dev_metrics, config):
    """Save adapters and head only; the frozen 7B backbone is never serialized."""
    parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    payload = {"parameters": parameters, "epoch": epoch, "dev_metrics": dev_metrics,
               "config": config, "target_modules": model.lora_target_modules}
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_trainable_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(payload["parameters"]) != set(expected) or payload["target_modules"] != model.lora_target_modules:
        raise ValueError("Checkpoint adapters/head do not match the current model.")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(payload["parameters"][name].to(device=parameter.device, dtype=parameter.dtype))
    return payload


def train_one_epoch(model, loader, optimizer, scheduler, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    device = next(model.backbone.parameters()).device
    loss_total = sample_total = window_samples = updates = 0
    for batch_index, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        size = len(batch["labels"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            loss = model(**batch)["loss"]
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite training loss.")
        # Sum per-example losses, then normalize at the optimizer boundary. This
        # also handles a short final batch/window without underweighting it.
        (loss * size).backward()
        loss_total += loss.item() * size
        sample_total += size
        window_samples += size
        if (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader):
            for parameter in trainable:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                    raise RuntimeError("Missing or nonfinite trainable gradient.")
                parameter.grad.div_(window_samples)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            window_samples = 0
            updates += 1
            if args.logging_steps and updates % args.logging_steps == 0:
                logging.info("Update %s, mean loss %.6f, lr %.8g", updates,
                             loss_total / sample_total, scheduler.get_last_lr()[0])
    return {"loss": loss_total / sample_total, "optimizer_steps": updates}


@torch.no_grad()
def evaluate_model(model, loader, bf16):
    model.eval()
    device = next(model.backbone.parameters()).device
    probabilities, labels = [], []
    total_loss = count = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            output = model(**batch)
        batch_probabilities = torch.softmax(output["logits"].float(), dim=-1)
        if model.num_labels == 2:
            batch_probabilities = batch_probabilities[:, 1]
        probabilities.extend(batch_probabilities.cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
        size = len(batch["labels"])
        total_loss += output["loss"].item() * size
        count += size
    metrics = calculate_metric_with_sklearn(
        probabilities, labels, class_labels=getattr(model, "class_labels", None),
    )
    metrics["loss"] = total_loss / count
    return metrics, probabilities, labels


def run_seed(model_args, args, dataset, seed, prepared_data=None, model_loader=None):
    """Train one independent seed, restore best dev epoch, and evaluate test once."""
    if not torch.cuda.is_available() or (args.bf16 and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Run training in a Slurm allocation with a bf16-capable GPU.")
    splits, dataset_info = prepared_data or prepare_dataset_data(dataset, args.model_max_length)
    recipe = asdict(args)
    for key in ("output_dir", "seed", "logging_steps", "num_workers"):
        recipe.pop(key)
    config = {"dataset_id": dataset.identifier, "seed": seed, "selection_metric": "loss",
              "selection_mode": "min",
              "model": {"name": "arcinstitute/evo2_7b", **asdict(model_args)},
              "recipe": {"lr_scheduler_type": "linear", **recipe}, "dataset": dataset_info}
    output_dir = dataset_output_dir(args.output_dir, dataset) / f"seed_{seed}"
    result_path = output_dir / "test_results.json"
    if result_path.exists():
        raise FileExistsError(f"Completed run already exists: {output_dir}. Use a new output_dir.")
    if (output_dir / "config.json").exists() and json.loads((output_dir / "config.json").read_text()) != config:
        raise FileExistsError(f"Existing run has a different configuration: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    seed_everything(seed)  # Before loading, classifier initialization, and adapter initialization.
    torch.cuda.reset_peak_memory_stats()
    backbone, tokenizer = (model_loader or load_evo2_backbone)(model_args)
    num_labels = dataset_info["num_labels"]
    model = configure_wqkv_lora(Evo2ForSequenceClassification(backbone, tokenizer.pad_id, num_labels), model_args)
    model.class_labels = dataset_info["class_labels"]
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in ("train", "dev", "test"):
        encoded_dataset = SupervisedDataset(None, tokenizer, args.model_max_length, rows=splits[split], num_labels=num_labels)
        loaders[split] = DataLoader(
            encoded_dataset, batch_size=args.per_device_train_batch_size if split == "train" else args.per_device_eval_batch_size,
            shuffle=split == "train", generator=generator if split == "train" else None,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer), num_workers=args.num_workers,
            pin_memory=True, drop_last=False,
        )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    history = []
    best_loss, best_epoch = math.inf, None
    for epoch in range(1, args.num_train_epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, scheduler, args)
        dev_metrics, _, _ = evaluate_model(model, loaders["dev"], args.bf16)
        history.append({"epoch": epoch, "train": train_metrics, "dev": dev_metrics})
        dev_loss = dev_metrics["loss"]
        if not math.isfinite(dev_loss):
            raise RuntimeError(f"Nonfinite dev loss for {dataset.identifier}, seed {seed}, epoch {epoch}.")
        logging.info("%s seed %s epoch %s: dev loss=%.6f", dataset.identifier, seed, epoch, dev_loss)
        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch
            save_trainable_checkpoint(model, output_dir / "best.pt", epoch, dev_metrics, config)
        write_json(output_dir / "history.json", history)
    checkpoint = load_trainable_checkpoint(model, output_dir / "best.pt")
    test_metrics, probabilities, labels = evaluate_model(model, loaders["test"], args.bf16)
    class_labels = dataset_info["class_labels"]
    with (output_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        if num_labels == 2:
            writer.writerow(["row_index", "label", "probability", "prediction"])
            writer.writerows((i, class_labels[label], probability, class_labels[int(probability >= 0.5)])
                             for i, (label, probability) in enumerate(zip(labels, probabilities)))
        else:
            writer.writerow(["row_index", "label", *[f"probability_{label}" for label in class_labels], "prediction"])
            writer.writerows((i, class_labels[label], *probability, class_labels[int(np.argmax(probability))])
                             for i, (label, probability) in enumerate(zip(labels, probabilities)))
    result = {"dataset": dataset.identifier, "dataset_name": dataset.name,
              "dataset_path": str(dataset.folder), "seed": seed, "selection_metric": "loss",
              "selection_mode": "min", "best_epoch": best_epoch,
              "best_dev": checkpoint["dev_metrics"], "test": test_metrics, "config": config,
              "schedule": {"optimizer_steps_per_epoch": steps_per_epoch, "total_optimizer_steps": total_steps,
                           "warmup_steps": warmup_steps},
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "gpu": torch.cuda.get_device_name(), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
    write_json(result_path, result)
    write_metric_details(output_dir, test_metrics)
    return result


def write_metric_details(folder, metrics):
    """Write human-readable per-class metrics."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if metrics.get("per_class"):
        with (folder / "test_per_class.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0]))
            writer.writeheader()
            writer.writerows(metrics["per_class"])


def dataset_output_dir(output_dir, dataset):
    root = Path(output_dir).resolve()
    path = (root / dataset.identifier).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Invalid dataset output identifier: {dataset.identifier!r}")
    return path


def train():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    datasets = resolve_datasets(data_args)
    for dataset in datasets:
        prepared = prepare_dataset_data(dataset, args.model_max_length)
        run_seed(model_args, args, dataset, args.seed, prepared_data=prepared)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
