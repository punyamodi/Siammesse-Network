import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall


def evaluate_model(model: tf.keras.Model, test_data: tf.data.Dataset) -> dict:
    recall = Recall()
    precision = Precision()

    all_y_true = []
    all_y_pred = []

    for test_input, test_val, y_true in test_data:
        yhat = model.predict([test_input, test_val], verbose=0)
        recall.update_state(y_true, yhat)
        precision.update_state(y_true, yhat)
        all_y_true.extend(y_true.numpy())
        all_y_pred.extend(yhat.flatten())

    r = recall.result().numpy()
    p = precision.result().numpy()
    f1 = 2 * p * r / (p + r + 1e-8)

    y_pred_binary = [1 if pred > 0.5 else 0 for pred in all_y_pred]
    accuracy = np.mean(np.array(y_pred_binary) == np.array(all_y_true))

    metrics = {
        "recall": float(r),
        "precision": float(p),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
    }

    print("\nEvaluation Results")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    return metrics
