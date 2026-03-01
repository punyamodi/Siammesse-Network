import argparse
import os
import sys
import json
import matplotlib.pyplot as plt

import tensorflow as tf

from siamese.config import (
    MODEL_SAVE_PATH,
    ANC_PATH,
    POS_PATH,
    NEG_PATH,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
)
from siamese.model import make_embedding, make_siamese_model, L1Dist
from siamese.dataset import build_dataset
from siamese.train import train
from siamese.evaluate import evaluate_model
from siamese.data_collection import collect_data
from siamese.verify import run_realtime_verification


def _setup_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def _load_model(model_path: str) -> tf.keras.Model:
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train first.")
        sys.exit(1)
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"L1Dist": L1Dist},
    )


def _plot_history(history: dict, save_path: str = "training_history.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["loss"])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")

    axes[1].plot(history["recall"], color="tab:orange")
    axes[1].set_title("Recall")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(history["precision"], color="tab:green")
    axes[2].set_title("Precision")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")


def cmd_collect(_args):
    collect_data()


def cmd_train(args):
    _setup_gpu()

    for path in [ANC_PATH, POS_PATH, NEG_PATH]:
        if not os.path.isdir(path) or len(os.listdir(path)) == 0:
            print(f"Directory {path} is missing or empty. Run 'collect' first and add LFW negatives.")
            sys.exit(1)

    print("Building dataset...")
    train_data, test_data = build_dataset(
        sample_size=args.samples,
        batch_size=args.batch_size,
        use_augmentation=args.augment,
    )

    embedding = make_embedding()
    model = make_siamese_model(embedding)
    model.summary()

    print("\nStarting training...")
    history = train(
        model=model,
        train_data=train_data,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_freq=args.checkpoint_freq,
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    _plot_history(history)

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nRunning evaluation on test set...")
    evaluate_model(model, test_data)


def cmd_evaluate(args):
    _setup_gpu()
    model = _load_model(args.model_path)

    print("Building dataset...")
    _, test_data = build_dataset(sample_size=args.samples, batch_size=args.batch_size)
    evaluate_model(model, test_data)


def cmd_verify(args):
    _setup_gpu()
    model = _load_model(args.model_path)
    run_realtime_verification(model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siamese",
        description="Siamese Network for one-shot face verification",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect", help="Collect anchor and positive images via webcam")

    train_p = subparsers.add_parser("train", help="Train the Siamese Network")
    train_p.add_argument("--samples", type=int, default=300, help="Images per class (default 300)")
    train_p.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default {EPOCHS})")
    train_p.add_argument("--batch-size", type=int, default=BATCH_SIZE, dest="batch_size")
    train_p.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    train_p.add_argument("--augment", action="store_true", help="Enable data augmentation")
    train_p.add_argument("--checkpoint-freq", type=int, default=10, dest="checkpoint_freq")

    eval_p = subparsers.add_parser("evaluate", help="Evaluate the trained model on test data")
    eval_p.add_argument("--model-path", default=MODEL_SAVE_PATH, dest="model_path")
    eval_p.add_argument("--samples", type=int, default=300)
    eval_p.add_argument("--batch-size", type=int, default=BATCH_SIZE, dest="batch_size")

    verify_p = subparsers.add_parser("verify", help="Run real-time face verification via webcam")
    verify_p.add_argument("--model-path", default=MODEL_SAVE_PATH, dest="model_path")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "collect": cmd_collect,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "verify": cmd_verify,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
