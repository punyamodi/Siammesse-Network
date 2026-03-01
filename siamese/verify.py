import os
import uuid
import numpy as np
import cv2
import tensorflow as tf

from siamese.config import (
    INPUT_IMAGE_PATH,
    VERIFICATION_IMAGES_PATH,
    DETECTION_THRESHOLD,
    VERIFICATION_THRESHOLD,
    WEBCAM_CAPTURE_SIZE,
    WEBCAM_OFFSET_X,
    WEBCAM_OFFSET_Y,
)
from siamese.dataset import preprocess


def verify(
    model: tf.keras.Model,
    detection_threshold: float = DETECTION_THRESHOLD,
    verification_threshold: float = VERIFICATION_THRESHOLD,
) -> tuple[list, bool]:
    results = []
    input_img = preprocess(os.path.join(INPUT_IMAGE_PATH, "input_image.jpg"))

    for image_name in os.listdir(VERIFICATION_IMAGES_PATH):
        validation_img = preprocess(
            os.path.join(VERIFICATION_IMAGES_PATH, image_name)
        )
        result = model.predict(
            [
                np.expand_dims(input_img, axis=0),
                np.expand_dims(validation_img, axis=0),
            ],
            verbose=0,
        )
        results.append(result[0][0])

    detection = int(np.sum(np.array(results) > detection_threshold))
    total_verification_images = len(os.listdir(VERIFICATION_IMAGES_PATH))
    verification_ratio = detection / (total_verification_images + 1e-8)
    verified = verification_ratio > verification_threshold

    return results, verified


def _crop_frame(frame):
    h, w = WEBCAM_CAPTURE_SIZE
    return frame[WEBCAM_OFFSET_Y:WEBCAM_OFFSET_Y + h, WEBCAM_OFFSET_X:WEBCAM_OFFSET_X + w, :]


def run_realtime_verification(model: tf.keras.Model):
    os.makedirs(INPUT_IMAGE_PATH, exist_ok=True)
    os.makedirs(VERIFICATION_IMAGES_PATH, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    print("Press 'v' to verify, 's' to save verification image, 'q' to quit.")
    last_status = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped = _crop_frame(frame)
        display = cropped.copy()

        status_color = (0, 255, 0) if last_status == "VERIFIED" else (0, 0, 255)
        cv2.putText(display, last_status, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(display, "[v=verify  s=save ref  q=quit]", (5, display.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.imshow("Face Verification", display)

        key = cv2.waitKey(10) & 0xFF

        if key == ord("v"):
            cv2.imwrite(os.path.join(INPUT_IMAGE_PATH, "input_image.jpg"), cropped)
            results, verified = verify(model)
            last_status = "VERIFIED" if verified else "NOT VERIFIED"
            positive_preds = int(np.sum(np.array(results) > DETECTION_THRESHOLD))
            print(f"Verified: {verified}  ({positive_preds}/{len(results)} positive predictions)")

        elif key == ord("s"):
            ref_path = os.path.join(VERIFICATION_IMAGES_PATH, f"{uuid.uuid1()}.jpg")
            cv2.imwrite(ref_path, cropped)
            print(f"Saved reference image: {ref_path}")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
