from pathlib import Path

import torch

from model import MoodForgeModel, load_model


def export_model():
    model, vocab, class_names = load_model()
    model.eval()

    sample = torch.zeros((1, len(vocab)), dtype=torch.float32)
    output_path = Path("moodforge.onnx")

    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["input_text_vector"],
        output_names=["style_logits"],
        dynamic_axes={
            "input_text_vector": {0: "batch_size"},
            "style_logits": {0: "batch_size"},
        },
    )

    print(f"Exported ONNX model to {output_path}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    export_model()
