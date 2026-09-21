import argparse

from model import load_model, predict_text


def parse_args():
    parser = argparse.ArgumentParser(description="MoodForge: a local custom AI model for style classification.")
    parser.add_argument("--train", action="store_true", help="Train the custom model on sample style data.")
    parser.add_argument("--predict", type=str, help="Predict the style of a sentence.")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive prediction mode.")
    parser.add_argument("--export-onnx", action="store_true", help="Export the trained model to ONNX for edge/NPU use.")
    return parser.parse_args()


def run_predict(text: str):
    model, vocab, class_names = load_model()
    result, confidence = predict_text(model, text, vocab, class_names)
    print(f"Predicted style: {result} ({confidence:.2%} confidence)")


def run_interactive():
    model, vocab, class_names = load_model()
    print("MoodForge is ready. Type 'exit' to quit.")
    while True:
        text = input("Text: ")
        if text.strip().lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        result, confidence = predict_text(model, text, vocab, class_names)
        print(f"Predicted style: {result} ({confidence:.2%} confidence)")


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        import train
        train.train_model()
    elif args.predict:
        run_predict(args.predict)
    elif args.interactive:
        run_interactive()
    elif args.export_onnx:
        import export_onnx
        export_onnx.export_model()
    else:
        print("Usage:")
        print("  python app.py --train")
        print("  python app.py --predict \"we need a bold launch plan\"")
        print("  python app.py --interactive")
        print("  python app.py --export-onnx")
