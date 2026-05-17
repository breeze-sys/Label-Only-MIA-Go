from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from classifier import CNN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_JSON_PATH = "shadow_train_data.json"
DEFAULT_OUTPUT_DIR = "python_server/CIFAR10/shadow_json_aligned"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ShadowJSONDataset(Dataset):
    def __init__(self, json_path: str, max_samples: int | None = None, seed: int = 42):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.source_indices = list(range(len(data)))
        if max_samples is not None and 0 < max_samples < len(data):
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(len(data)), max_samples))
            data = [data[i] for i in indices]
            self.source_indices = indices

        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        image = item.get("image", item.get("Image"))
        label = item.get("target_label", item.get("Label"))
        if image is None or label is None:
            raise KeyError("JSON item must contain image/Image and target_label/Label")

        soft_target = (
            item.get("soft_label")
            or item.get("soft_labels")
            or item.get("probabilities")
            or item.get("probs")
            or item.get("logits")
        )

        image_tensor = torch.tensor(image, dtype=torch.float32).view(3, 32, 32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        if soft_target is None:
            soft_target_tensor = torch.empty(0, dtype=torch.float32)
        else:
            soft_target_tensor = torch.tensor(soft_target, dtype=torch.float32)
        return image_tensor, label_tensor, soft_target_tensor


def batch_loss(logits: torch.Tensor, labels: torch.Tensor, soft_targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    if soft_targets.ndim == 2 and soft_targets.size(1) == logits.size(1):
        normalized = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_probs = F.log_softmax(logits, dim=1)
        return -(normalized * log_probs).sum(dim=1).mean()
    return criterion(logits, labels)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_raw_loss: float
    train_acc: float
    train_confidence: float
    val_loss: float
    val_raw_loss: float
    val_acc: float
    val_confidence: float


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_raw_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_confidence = 0.0

    with torch.no_grad():
        for images, labels, soft_targets in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            soft_targets = soft_targets.to(DEVICE)

            logits = model(images)
            loss = batch_loss(logits, labels, soft_targets, criterion)
            raw_loss = F.cross_entropy(logits, labels)
            probs = torch.softmax(logits, dim=1)
            confidences, preds = probs.max(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_raw_loss += raw_loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_confidence += confidences.sum().item()
            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0, 0.0

    return (
        total_loss / total_samples,
        total_raw_loss / total_samples,
        total_correct / total_samples,
        total_confidence / total_samples,
    )


def train_json_shadow(args: argparse.Namespace) -> None:
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"shadow train json not found: {args.json_path}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = ShadowJSONDataset(args.json_path, max_samples=args.max_samples, seed=args.seed)
    if len(dataset) < 2:
        raise ValueError("dataset is too small to train")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("train split is empty; lower val_ratio or increase max_samples")

    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_source_indices = [dataset.source_indices[i] for i in train_set.indices]
    val_source_indices = [dataset.source_indices[i] for i in val_set.indices]

    model = CNN(args.model_arch, args.dataset, dropout=True, dropout_p=args.dropout_p).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_checkpoint_path = os.path.join(args.output_dir, "best_checkpoint_ep.pth")
    last_checkpoint_path = os.path.join(args.output_dir, "last_checkpoint_ep.pth")
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    train_indices_path = os.path.join(args.output_dir, "train_indices.json")
    val_indices_path = os.path.join(args.output_dir, "val_indices.json")

    metrics_rows: list[EpochMetrics] = []
    best_score = float("inf")
    best_epoch = 0
    no_improve_epochs = 0

    print(f"[*] Start JSON-aligned shadow training on {DEVICE}")
    print(f"[*] Samples: total={len(dataset)}, train={len(train_set)}, val={len(val_set)}")
    print(
        "[*] Hyperparams: "
        f"epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}, "
        f"dropout={args.dropout_p}, target_loss={args.target_loss}, hard_stop_loss={args.hard_stop_loss}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_raw_loss = 0.0
        running_correct = 0
        running_samples = 0
        running_confidence = 0.0

        for images, labels, soft_targets in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            soft_targets = soft_targets.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = batch_loss(logits, labels, soft_targets, criterion)
            raw_loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                confidences, preds = probs.max(dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_raw_loss += raw_loss.item() * batch_size
            running_correct += (preds == labels).sum().item()
            running_confidence += confidences.sum().item()
            running_samples += batch_size

        train_loss = running_loss / running_samples
        train_raw_loss = running_raw_loss / running_samples
        train_acc = running_correct / running_samples
        train_conf = running_confidence / running_samples
        val_loss, val_raw_loss, val_acc, val_conf = evaluate(model, val_loader, criterion)

        row = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_raw_loss=train_raw_loss,
            train_acc=train_acc,
            train_confidence=train_conf,
            val_loss=val_loss,
            val_raw_loss=val_raw_loss,
            val_acc=val_acc,
            val_confidence=val_conf,
        )
        metrics_rows.append(row)

        score = abs(train_raw_loss - args.target_loss) + 0.3 * abs(val_raw_loss - args.target_loss)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "train_loss": train_loss,
                    "train_raw_loss": train_raw_loss,
                    "train_acc": train_acc,
                    "train_confidence": train_conf,
                    "val_loss": val_loss,
                    "val_raw_loss": val_raw_loss,
                    "val_acc": val_acc,
                    "val_confidence": val_conf,
                    "config": vars(args),
                },
                best_checkpoint_path,
            )
        else:
            no_improve_epochs += 1

        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "train_loss": train_loss,
                "train_raw_loss": train_raw_loss,
                "val_loss": val_loss,
                "val_raw_loss": val_raw_loss,
                "config": vars(args),
            },
            last_checkpoint_path,
        )

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} raw={train_raw_loss:.4f} "
            f"train_acc={train_acc:.4f} train_conf={train_conf:.4f} | "
            f"val_loss={val_loss:.4f} raw={val_raw_loss:.4f} "
            f"val_acc={val_acc:.4f} val_conf={val_conf:.4f}"
        )

        if train_raw_loss <= args.hard_stop_loss:
            print(
                f"[+] Hard stop triggered: train_raw_loss={train_raw_loss:.4f} "
                f"<= {args.hard_stop_loss:.4f}"
            )
            break

        if no_improve_epochs >= args.patience:
            print(f"[+] Early stop triggered after {args.patience} epochs without better score")
            break

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics_rows[0]).keys()))
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(asdict(row))

    with open(train_indices_path, "w", encoding="utf-8") as f:
        json.dump(train_source_indices, f)
    with open(val_indices_path, "w", encoding="utf-8") as f:
        json.dump(val_source_indices, f)

    summary = {
        "device": DEVICE,
        "dataset_size": len(dataset),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "best_epoch": best_epoch,
        "best_checkpoint": best_checkpoint_path,
        "last_checkpoint": last_checkpoint_path,
        "train_indices": train_indices_path,
        "val_indices": val_indices_path,
        "target_loss": args.target_loss,
        "hard_stop_loss": args.hard_stop_loss,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "max_samples": args.max_samples,
    }

    with open(os.path.join(args.output_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[+] JSON-aligned shadow model training finished")
    print(f"[+] Best checkpoint: {best_checkpoint_path}")
    print(f"[+] Metrics: {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LabelScan-Go shadow model training entry")
    parser.add_argument("--action", type=int, required=True, help="Use --action 7 for JSON-aligned shadow training")
    parser.add_argument("--json-path", default=DEFAULT_JSON_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--model-arch", default="CNN7")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.15)
    parser.add_argument("--target-loss", type=float, default=0.43)
    parser.add_argument("--hard-stop-loss", type=float, default=0.40)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.action != 7:
        raise SystemExit("Only --action 7 is implemented in this branch")

    train_json_shadow(args)


if __name__ == "__main__":
    main()
