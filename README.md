# Siamese Network — One-Shot Face Verification

A production-ready implementation of a Siamese Neural Network for one-shot face verification, built with TensorFlow and OpenCV. The system learns to distinguish whether two face images belong to the same person using L1 distance between deep embeddings.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## How It Works

A Siamese Network passes two images through the **same** convolutional embedding network (shared weights) to produce two feature vectors. The L1 distance between those vectors feeds a binary classifier that outputs a similarity score between 0 and 1.

```
Input Image ──┐
              ├──► Embedding CNN ──► L1 Distance ──► Dense(512) ──► Dense(1, sigmoid) ──► Score
Verify Image ─┘         (shared weights)
```

**Architecture of the Embedding Network:**

| Block | Layer                     | Output Shape     |
|-------|---------------------------|------------------|
| 1     | Conv2D(64, 10×10) + BN    | 96 × 96 × 64     |
| 1     | MaxPooling2D              | 48 × 48 × 64     |
| 2     | Conv2D(128, 7×7) + BN     | 42 × 42 × 128    |
| 2     | MaxPooling2D              | 21 × 21 × 128    |
| 3     | Conv2D(128, 4×4) + BN     | 18 × 18 × 128    |
| 3     | MaxPooling2D              | 9 × 9 × 128      |
| 4     | Conv2D(256, 4×4)          | 6 × 6 × 256      |
| 5     | Flatten → Dense(4096)     | 4096             |

---

## Project Structure

```
siamese-network/
├── main.py                    # CLI entry point
├── requirements.txt
├── siamese/
│   ├── config.py              # All paths and hyperparameters
│   ├── model.py               # Embedding CNN, L1Dist layer, Siamese model
│   ├── dataset.py             # Data loading, preprocessing, augmentation, splits
│   ├── train.py               # Training loop with checkpointing
│   ├── evaluate.py            # Precision, Recall, F1, Accuracy
│   ├── verify.py              # Real-time webcam verification
│   └── data_collection.py     # Webcam data collection utility
├── data/
│   ├── anchor/                # Anchor face images (your face)
│   ├── positive/              # Positive face images (your face, different shots)
│   └── negative/              # Negative face images (LFW dataset)
└── application_data/
    ├── input_image/           # Captured frame for verification
    └── verification_images/   # Reference images for verification
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get negative samples (LFW dataset)

Download the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset and extract images into `data/negative/`.

```bash
tar -xf lfw.tgz
python -c "
import os, shutil
for person in os.listdir('lfw'):
    for img in os.listdir(os.path.join('lfw', person)):
        shutil.move(os.path.join('lfw', person, img), os.path.join('data', 'negative', img))
"
```

### 3. Collect your face images

```bash
python main.py collect
```

- Press **`a`** to capture an anchor image
- Press **`p`** to capture a positive image
- Press **`q`** to quit

Aim for at least **300 images** of each type.

### 4. Train

```bash
python main.py train --epochs 50 --samples 300 --augment
```

| Flag              | Default | Description                        |
|-------------------|---------|------------------------------------|
| `--epochs`        | 50      | Number of training epochs          |
| `--samples`       | 300     | Images per class to use            |
| `--batch-size`    | 16      | Batch size                         |
| `--lr`            | 1e-4    | Learning rate                      |
| `--augment`       | off     | Enable random flip/brightness/contrast augmentation |
| `--checkpoint-freq` | 10   | Save checkpoint every N epochs     |

Training produces `siamese_model.keras`, `training_history.json`, and `training_history.png`.

### 5. Evaluate

```bash
python main.py evaluate
```

Reports Accuracy, Precision, Recall, and F1 Score on the held-out test set.

### 6. Real-time verification

```bash
python main.py verify
```

- Press **`s`** to save the current frame as a reference verification image
- Press **`v`** to run face verification against your reference images
- Press **`q`** to quit

---

## Configuration

All paths and hyperparameters are in `siamese/config.py`. Key settings:

| Variable                  | Default         | Description                                          |
|---------------------------|-----------------|------------------------------------------------------|
| `IMG_SIZE`                | (105, 105)      | Input image dimensions                               |
| `EMBEDDING_DIM`           | 4096            | Size of the final embedding vector                   |
| `EPOCHS`                  | 50              | Default training epochs                              |
| `LEARNING_RATE`           | 1e-4            | Adam optimizer learning rate                         |
| `TRAIN_SPLIT`             | 0.7             | Fraction of data used for training                   |
| `DETECTION_THRESHOLD`     | 0.5             | Minimum similarity score to count as a match         |
| `VERIFICATION_THRESHOLD`  | 0.6             | Minimum proportion of matches to verify identity     |

---

## Features

- Modular Python package — no notebooks required
- Shared-weight embedding CNN with Batch Normalization
- L1 distance layer as a custom Keras layer (serializable)
- Data augmentation (flip, brightness, contrast)
- Custom training loop with gradient tape
- Checkpoint saving every N epochs
- Full evaluation suite: Accuracy, Precision, Recall, F1
- Real-time webcam verification with on-screen status
- CLI with subcommands: `collect`, `train`, `evaluate`, `verify`
- GPU memory growth configuration out of the box

---

## References

- [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) — Koch et al., 2015
- [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
