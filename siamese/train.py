import os
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

from siamese.config import CHECKPOINT_DIR, LEARNING_RATE, EPOCHS


def _train_step(
    batch,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        yhat = model(X, training=True)
        loss = loss_fn(y, yhat)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train(
    model: tf.keras.Model,
    train_data: tf.data.Dataset,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    checkpoint_freq: int = 10,
) -> dict:
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.losses.BinaryCrossentropy()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    history = {"loss": [], "recall": [], "precision": []}

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        progbar = tf.keras.utils.Progbar(len(train_data))

        recall = Recall()
        precision = Precision()
        epoch_losses = []

        for idx, batch in enumerate(train_data):
            loss = _train_step(batch, model, optimizer, loss_fn)
            epoch_losses.append(loss.numpy())

            yhat = model.predict(batch[:2], verbose=0)
            recall.update_state(batch[2], yhat)
            precision.update_state(batch[2], yhat)
            progbar.update(idx + 1)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        r = recall.result().numpy()
        p = precision.result().numpy()

        history["loss"].append(avg_loss)
        history["recall"].append(r)
        history["precision"].append(p)

        print(f"  loss={avg_loss:.4f}  recall={r:.4f}  precision={p:.4f}")

        if epoch % checkpoint_freq == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"  Checkpoint saved at epoch {epoch}")

    return history
