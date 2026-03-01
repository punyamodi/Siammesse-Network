import os
import uuid
import cv2

from siamese.config import (
    ANC_PATH,
    POS_PATH,
    WEBCAM_CAPTURE_SIZE,
    WEBCAM_OFFSET_X,
    WEBCAM_OFFSET_Y,
)


def _crop_frame(frame):
    h, w = WEBCAM_CAPTURE_SIZE
    return frame[WEBCAM_OFFSET_Y:WEBCAM_OFFSET_Y + h, WEBCAM_OFFSET_X:WEBCAM_OFFSET_X + w, :]


def collect_data():
    os.makedirs(ANC_PATH, exist_ok=True)
    os.makedirs(POS_PATH, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    print("Press 'a' to capture anchor, 'p' to capture positive, 'q' to quit.")
    anchor_count = 0
    positive_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped = _crop_frame(frame)
        display = cropped.copy()
        cv2.putText(display, f"Anchors: {anchor_count}  Positives: {positive_count}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Data Collection  [a=anchor  p=positive  q=quit]", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):
            path = os.path.join(ANC_PATH, f"{uuid.uuid1()}.jpg")
            cv2.imwrite(path, cropped)
            anchor_count += 1

        elif key == ord("p"):
            path = os.path.join(POS_PATH, f"{uuid.uuid1()}.jpg")
            cv2.imwrite(path, cropped)
            positive_count += 1

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {anchor_count} anchor images and {positive_count} positive images.")
