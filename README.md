# Siamese Network — One-Shot Face Verification

> Verify identity from a single reference image using deep metric learning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Most face recognition systems require hundreds of labeled photos per person. **One-shot learning** sidesteps this: the model learns a similarity function rather than fixed identity classes, so it can verify any person from a single reference image — even identities it has never seen during training.

This project implements a Siamese Network following [Koch et al., 2015](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), with a complete pipeline from webcam data collection through training, evaluation, and real-time verification.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Collect face images](#1-collect-face-images)
  - [2. Add negative samples](#2-add-negative-samples-lfw)
  - [3. Train](#3-train)
  - [4. Evaluate](#4-evaluate)
  - [5. Real-time verification](#5-real-time-verification)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [References](#references)

---

## How It Works

A Siamese Network runs two images through the **exact same** convolutional network (shared weights). The resulting feature vectors are compared with an L1 distance layer, and a small classifier decides if the images show the same person.

```
                    ┌─────────────────────────────────────────────┐
  Input Image  ───► │                                             │
                    │   Conv Block × 4  →  Flatten  →  Dense(4096)│  ─► embedding A
                    │       (shared weights)                      │
  Verify Image ───► │                                             │  ─► embedding B
                    └─────────────────────────────────────────────┘
                                          │
                               L1 Distance  |A - B|
                                          │
                               Dense(512, relu)
                                          │
                               Dense(1, sigmoid)
                                          │
                                   Score ∈ [0, 1]
                              0 = different person
                              1 = same person
```

**Verification logic:**  
The system compares a live webcam frame against a set of stored reference images. If more than `VERIFICATION_THRESHOLD` (default 60%) of comparisons score above `DETECTION_THRESHOLD` (default 0.5), identity is confirmed.

---

## Architecture

### Embedding Network

| Block | Operation               | Output Shape    |
|-------|-------------------------|-----------------|
| 1     | Conv2D(64, 10×10) + BN  | 96 × 96 × 64    |
|       | MaxPooling2D(2×2)       | 48 × 48 × 64    |
| 2     | Conv2D(128, 7×7) + BN   | 42 × 42 × 128   |
|       | MaxPooling2D(2×2)       | 21 × 21 × 128   |
| 3     | Conv2D(128, 4×4) + BN   | 18 × 18 × 128   |
|       | MaxPooling2D(2×2)       | 9 × 9 × 128     |
| 4     | Conv2D(256, 4×4)        | 6 × 6 × 256     |
|       | Flatten → Dense(4096)   | 4096            |

### Siamese Head

```
L1Distance(embedding_A, embedding_B)  →  Dense(512, relu)  →  Dropout(0.3)  →  Dense(1, sigmoid)
```

Batch Normalization after each convolutional block stabilizes training and allows a higher learning rate. The Dropout in the head prevents the classifier from overfitting once embeddings are well-separated.

---

## Project Structure

```
Siammesse-Network/
├── main.py                      CLI entry point (collect / train / evaluate / verify)
├── requirements.txt
├── siamese/
│   ├── config.py                All paths and hyperparameters in one place
│   ├── model.py                 Embedding CNN, L1Dist layer, Siamese model factory
│   ├── dataset.py               tf.data pipeline: load, preprocess, augment, split
│   ├── train.py                 Custom gradient-tape training loop + checkpointing
│   ├── evaluate.py              Accuracy, Precision, Recall, F1 over test set
│   ├── data_collection.py       Webcam utility — capture anchor and positive images
│   └── verify.py                Real-time webcam verification with on-screen result
├── data/
│   ├── anchor/                  Reference images of the person to verify (your face)
│   ├── positive/                Additional shots of the same person
│   └── negative/                Impostor images (populated from LFW)
└── application_data/
    ├── input_image/             Live captured frame used at verification time
    └── verification_images/     Stored reference images compared against input
```

---

## Setup

**Python 3.10+ required.**

```bash
git clone https://github.com/punyamodi/Siammesse-Network.git
cd Siammesse-Network
pip install -r requirements.txt
```

GPU is optional. If a CUDA-capable GPU is present, TensorFlow will use it automatically; memory growth is configured at startup to prevent OOM errors.

---

## Usage

### 1. Collect face images

```bash
python main.py collect
```

A webcam window opens. While keeping your face centered in the frame:

| Key | Action |
|-----|--------|
| `a` | Save current frame as an **anchor** image |
| `p` | Save current frame as a **positive** image |
| `q` | Quit collection |

Capture at least **300 anchor** and **300 positive** images. Vary lighting, distance, and expression for a more robust model.

Images are saved to `data/anchor/` and `data/positive/` with UUID filenames.

---

### 2. Add negative samples (LFW)

Download the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset, then flatten all images into `data/negative/`:

```bash
# Download from http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xf lfw.tgz

python - <<'EOF'
import os, shutil
os.makedirs(os.path.join('data', 'negative'), exist_ok=True)
for person in os.listdir('lfw'):
    for img in os.listdir(os.path.join('lfw', person)):
        src = os.path.join('lfw', person, img)
        dst = os.path.join('data', 'negative', img)
        shutil.move(src, dst)
EOF
```

LFW contains ~13,000 images of public figures — none of them you — making it an ideal negative set.

---

### 3. Train

```bash
python main.py train --epochs 50 --samples 300 --augment
```

Training outputs:

| File | Contents |
|------|----------|
| `siamese_model.keras` | Saved model (weights + architecture) |
| `training_history.json` | Per-epoch loss, recall, precision |
| `training_history.png` | Loss / recall / precision curves |
| `training_checkpoints/` | Periodic weight snapshots |

**All training flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Training epochs |
| `--samples` | 300 | Images per class |
| `--batch-size` | 16 | Batch size |
| `--lr` | 1e-4 | Adam learning rate |
| `--augment` | off | Random flip, brightness, contrast jitter |
| `--checkpoint-freq` | 10 | Save checkpoint every N epochs |

---

### 4. Evaluate

```bash
python main.py evaluate
```

Runs the saved model over the held-out test split and prints:

```
Evaluation Results
  Accuracy:  0.9650
  Precision: 0.9712
  Recall:    0.9583
  F1 Score:  0.9647
```

Pass `--model-path path/to/model.keras` to evaluate a specific checkpoint.

---

### 5. Real-time verification

```bash
python main.py verify
```

A webcam window opens showing the live feed. First build a reference set, then verify:

| Key | Action |
|-----|--------|
| `s` | Save current frame to `application_data/verification_images/` as a reference |
| `v` | Capture input frame and run verification against all reference images |
| `q` | Quit |

The window displays **VERIFIED** (green) or **NOT VERIFIED** (red) after each verification attempt. The console prints the raw match count for debugging.

---

## Configuration

Edit `siamese/config.py` to change any default. All modules import from there — no magic numbers elsewhere.

| Variable | Default | Description |
|----------|---------|-------------|
| `IMG_SIZE` | `(105, 105)` | Resize target for all input images |
| `EMBEDDING_DIM` | `4096` | Output dimension of the embedding network |
| `BATCH_SIZE` | `16` | Training and evaluation batch size |
| `EPOCHS` | `50` | Default number of training epochs |
| `LEARNING_RATE` | `1e-4` | Adam optimizer learning rate |
| `TRAIN_SPLIT` | `0.7` | Fraction of data reserved for training |
| `DETECTION_THRESHOLD` | `0.5` | Minimum score to count one comparison as a match |
| `VERIFICATION_THRESHOLD` | `0.6` | Minimum match ratio across all reference images to confirm identity |

---

## CLI Reference

```
usage: python main.py <command> [options]

commands:
  collect     Capture anchor and positive face images via webcam
  train       Train the Siamese Network
  evaluate    Evaluate the saved model on the test split
  verify      Run real-time face verification via webcam
```

```bash
python main.py train --help
python main.py evaluate --model-path siamese_model.keras --samples 300
python main.py verify --model-path siamese_model.keras
```

---

## References

- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf). ICML Deep Learning Workshop.
- Huang, G. B., et al. (2007). [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/). UMass Amherst Technical Report.
