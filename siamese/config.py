import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
POS_PATH = os.path.join(DATA_DIR, "positive")
NEG_PATH = os.path.join(DATA_DIR, "negative")
ANC_PATH = os.path.join(DATA_DIR, "anchor")

APP_DIR = os.path.join(BASE_DIR, "application_data")
INPUT_IMAGE_PATH = os.path.join(APP_DIR, "input_image")
VERIFICATION_IMAGES_PATH = os.path.join(APP_DIR, "verification_images")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "training_checkpoints")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "siamese_model.keras")

IMG_SIZE = (105, 105)
IMG_CHANNELS = 3

EMBEDDING_DIM = 4096
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.7

DETECTION_THRESHOLD = 0.5
VERIFICATION_THRESHOLD = 0.6

WEBCAM_CAPTURE_SIZE = (250, 250)
WEBCAM_OFFSET_X = 200
WEBCAM_OFFSET_Y = 120
