import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


MODEL_PATH = Path("moodforge_model.pt")
VOCAB_PATH = Path("moodforge_vocab.json")


class MoodForgeModel(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_vocab(sentences: dict[str, list[str]]) -> dict[str, int]:
    vocab = {"<PAD>": 0}
    for texts in sentences.values():
        for text in texts:
            for token in text.lower().split():
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def text_to_vector(text: str, vocab: dict[str, int]) -> np.ndarray:
    counts = np.zeros(len(vocab), dtype=np.float32)
    for token in text.lower().split():
        idx = vocab.get(token)
        if idx is not None:
            counts[idx] += 1.0
    return counts


def prepare_dataset(sentences: dict[str, list[str]], vocab: dict[str, int]):
    labels = []
    features = []
    label_to_index = {label: idx for idx, label in enumerate(sentences.keys())}

    for label, texts in sentences.items():
        for text in texts:
            features.append(text_to_vector(text, vocab))
            labels.append(label_to_index[label])

    x = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y, label_to_index


def save_model(model: MoodForgeModel, vocab: dict[str, int], class_names: list[str]):
    torch.save({
        "model_state": model.state_dict(),
        "class_names": class_names,
        "vocab": vocab,
    }, MODEL_PATH)
    with VOCAB_PATH.open("w", encoding="utf-8") as handle:
        json.dump({"vocab": vocab, "class_names": class_names}, handle, ensure_ascii=False, indent=2)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train it first with: python train.py")

    payload = torch.load(MODEL_PATH, map_location="cpu")
    vocab = payload["vocab"]
    class_names = payload["class_names"]
    model = MoodForgeModel(len(vocab), len(class_names))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, vocab, class_names


def predict_text(model: MoodForgeModel, text: str, vocab: dict[str, int], class_names: list[str]):
    vector = torch.tensor(text_to_vector(text, vocab), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(vector)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    return class_names[int(idx)], float(conf)
