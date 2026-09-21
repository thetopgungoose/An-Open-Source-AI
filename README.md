MoodForge is a compact local AI model for classifying text by tone and style. It is designed as a realistic custom-model project for a Windows machine with an NPU, GPU, or CPU. 

This is not a generic chatbot. It is a custom model trained to recognize the writing style of a sentence across four categories:

- calm
- bold
- creative
- technical

That makes it useful for:

- writing style detection
- assistant personality tuning
- custom moderation rules
- local personal AI behavior
- NPU-friendly edge inference experiments

## Model design

The architecture is intentionally light and portable:

- bag-of-words text vectorization
- a small PyTorch feed-forward classifier
- ONNX export support for edge/NPU deployment

This keeps the project fast, understandable, and easy to extend.

## Install

```bash
cd open-source-ai
python -m venv .venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

## Predict a style

```bash
python app.py --predict "we should launch with bold confidence"
```

## Interactive mode

```bash
python app.py --interactive
```

## Export to ONNX for NPU / edge deployment

```bash
python app.py --export-onnx
```

This exports a file named `moodforge.onnx` that can be used in optimized runtimes or SDKs that support edge/NPU acceleration.

## Example output

```text
Predicted style: bold (93.12% confidence)
```

## Project files

- `model.py` — model definition and inference helpers
- `train.py` — custom dataset and training loop
- `app.py` — CLI interface
- `export_onnx.py` — ONNX export script

## Next upgrades

- add a bigger custom dataset
- use embeddings or a tiny transformer
- build a local web UI
- add multilingual text support
- integrate with Intel/AMD/NPU acceleration frameworks

This is a practical starter for building your own AI model locally instead of depending only on public hosted models.
