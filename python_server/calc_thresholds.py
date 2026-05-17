import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from classifier import CNN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JSON_PATH = os.getenv("SHADOW_JSON_PATH", "shadow_train_data.json")
SHADOW_MODEL_PATH = os.getenv(
    "SHADOW_MODEL_PATH",
    "python_server/CIFAR10/shadow_json_aligned/best_checkpoint_ep.pth",
)
SHADOW_TRAIN_INDICES_PATH = os.getenv("SHADOW_TRAIN_INDICES_PATH", "")
SHADOW_CONFIG_OUTPUT = os.getenv("SHADOW_CONFIG_OUTPUT", "shadow_config.json")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
MODEL_ARCH = os.getenv("MODEL_ARCH", "CNN7")
DATASET_NAME = os.getenv("DATASET_NAME", "CIFAR10")
MODEL_DROPOUT_P = float(os.getenv("MODEL_DROPOUT_P", "0.5"))
SHADOW_RED_QUANTILE = float(os.getenv("SHADOW_RED_QUANTILE", "40"))


class ShadowJSONDataset(Dataset):
    def __init__(self, json_path, indices_path=""):
        print(f"[*] Loading JSON dataset: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.source_count = len(data)
        self.indices_path = indices_path
        if indices_path:
            print(f"[*] Restricting threshold data to training indices: {indices_path}")
            with open(indices_path, "r", encoding="utf-8") as f:
                indices = json.load(f)
            data = [data[i] for i in indices]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item.get("image", item.get("Image"))
        label = item.get("target_label", item.get("Label"))
        if image is None or label is None:
            raise KeyError("JSON item must contain image/Image and target_label/Label")
        image_tensor = torch.tensor(image, dtype=torch.float32).view(3, 32, 32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


def load_state_dict(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    clean_ckpt = {}
    for k, v in ckpt.items():
        clean_ckpt[k[7:] if k.startswith("module.") else k] = v
    return clean_ckpt


def get_losses(model, loader):
    model.eval()
    losses = []
    confidences = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            logits = model(data)
            batch_losses = F.cross_entropy(logits, target, reduction="none")
            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)

            losses.extend(batch_losses.cpu().numpy().tolist())
            confidences.extend(conf.cpu().numpy().tolist())
            correct += (preds == target).sum().item()
            total += target.size(0)

    return (
        np.array(losses, dtype=np.float64),
        np.array(confidences, dtype=np.float64),
        correct / total if total else 0.0,
    )


def main():
    if not os.path.exists(SHADOW_MODEL_PATH):
        raise FileNotFoundError(f"shadow model not found: {SHADOW_MODEL_PATH}")
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"shadow train json not found: {JSON_PATH}")

    print(f"[*] Loading aligned shadow model from {SHADOW_MODEL_PATH}")
    model = CNN(MODEL_ARCH, DATASET_NAME, dropout=True, dropout_p=MODEL_DROPOUT_P).to(DEVICE)
    model.load_state_dict(load_state_dict(SHADOW_MODEL_PATH))

    dataset = ShadowJSONDataset(JSON_PATH, SHADOW_TRAIN_INDICES_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    losses, confidences, accuracy = get_losses(model, loader)
    if len(losses) == 0:
        raise ValueError("no samples were available for threshold generation")
    if SHADOW_RED_QUANTILE < 0 or SHADOW_RED_QUANTILE > 100:
        raise ValueError("SHADOW_RED_QUANTILE must be between 0 and 100")

    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))
    threshold = float(np.percentile(losses, SHADOW_RED_QUANTILE))
    tau_opt = mean_loss
    median_loss = float(np.median(losses))
    p95_loss = float(np.percentile(losses, 95))
    mean_conf = float(np.mean(confidences))

    result = {
        "threshold": threshold,
        "mean_member_loss": mean_loss,
        "std_member_loss": std_loss,
        "tau_95": threshold,
        "tau_opt": tau_opt,
        "median_member_loss": median_loss,
        "p05_member_loss": float(np.percentile(losses, 5)),
        "p10_member_loss": float(np.percentile(losses, 10)),
        "p25_member_loss": float(np.percentile(losses, 25)),
        "p75_member_loss": float(np.percentile(losses, 75)),
        "p90_member_loss": float(np.percentile(losses, 90)),
        "p95_member_loss": p95_loss,
        "threshold_quantile": SHADOW_RED_QUANTILE,
        "mean_member_confidence": mean_conf,
        "member_accuracy": float(accuracy),
        "evaluated_sample_count": len(dataset),
        "source_sample_count": dataset.source_count,
        "shadow_json_path": JSON_PATH,
        "shadow_model_path": SHADOW_MODEL_PATH,
        "shadow_train_indices_path": SHADOW_TRAIN_INDICES_PATH,
        "model_arch": MODEL_ARCH,
        "dataset_name": DATASET_NAME,
        "model_dropout_p": MODEL_DROPOUT_P,
        "device": DEVICE,
        "description": "Generated by calc_thresholds.py (JSON aligned shadow model)",
    }

    with open(SHADOW_CONFIG_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("[+] shadow_config.json generated")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
