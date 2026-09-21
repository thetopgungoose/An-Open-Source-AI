# MoodForge, a relatively ethical AI with less environmental impact#
**Relatively early in development**

MoodForge is a compact local AI model for classifying text by tone and style. It is designed as a realistic custom-model project for a Windows machine with an NPU, like those that have Ryzen AI cpus, or intel Core Ultra 200V and 200S Series cpus.

I made it a local ai model due to my dislike of the environmental impact of data centers. If you need clankers so bad just use local ai.

This is not a generic chatbot. It is a custom model trained to recognize the writing style of a sentence across four categories, I will make changes to this when I have time. 

I purposely rendered this AI model to be incapable of image generation due to the fact that weirdoes love using it to do horrendous things. 

Some personalities it has:
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
