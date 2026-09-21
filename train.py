from pathlib import Path

import torch

from model import MoodForgeModel, build_vocab, prepare_dataset, save_model


TRAINING_DATA = {
    "calm": [
        "take a slow breath and think clearly",
        "let us focus on the next simple step",
        "this approach feels steady and measured",
        "we can move gently and keep it relaxed",
        "be patient and let the solution settle naturally",
        "a calm plan keeps the team centered and confident",
    ],
    "bold": [
        "we should go all in and move fast",
        "this idea deserves a fearless launch",
        "charge ahead with bold confidence",
        "we need decisive action now",
        "make the leap and push the vision forward",
        "take the risk and lead with conviction",
    ],
    "creative": [
        "imagine a bright idea that feels magical",
        "let the concept bloom with wonder and color",
        "this is playful, artistic, and imaginative",
        "build a fresh vision with spark and surprise",
        "we can paint the future with unusual energy",
        "turn the concept into something luminous and unexpected",
    ],
    "technical": [
        "optimize the pipeline for low latency",
        "we should measure throughput and benchmark reliability",
        "this requires structured logging and performance metrics",
        "focus on architecture, inference, and system design",
        "the system should minimize overhead and improve throughput",
        "profile the model and track latency across the stack",
    ],
}


def train_model():
    vocab = build_vocab(TRAINING_DATA)
    inputs, targets, _ = prepare_dataset(TRAINING_DATA, vocab)
    class_names = list(TRAINING_DATA.keys())

    model = MoodForgeModel(len(vocab), len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(350):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 349:
            print(f"epoch {epoch:03d} | loss {loss.item():.4f}")

    save_model(model, vocab, class_names)
    print(f"Saved model to {Path('moodforge_model.pt')}")
    print(f"Saved vocab to {Path('moodforge_vocab.json')}")
    print("Training complete. Use: python app.py --predict 'your text here'")


if __name__ == "__main__":
    train_model()
