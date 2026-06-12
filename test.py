import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import math
import numpy as np
import cv2
from tkinter import messagebox
from mtcnn import MTCNN
from tensorflow.keras import models

MODEL_NAME = "resnet18"
MODEL_PATH = f"./{MODEL_NAME}.h5"
FOLDER_PATH = "database/train"
INPUT_SIZE = (224, 224)
THRESHOLD = 0.7
CAMERA_ID = 0

model = models.load_model(MODEL_PATH)
cls_list = sorted(
    d for d in os.listdir(FOLDER_PATH) if os.path.isdir(os.path.join(FOLDER_PATH, d))
)
print("類別:", cls_list)

detector = MTCNN()


def align_face(img_rgb, keypoints, box, out_size=INPUT_SIZE, scale=2.2):
    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img_rgb, M, (img_rgb.shape[1], img_rgb.shape[0]))
    aspect = out_size[0] / out_size[1]
    box_h = h * scale
    box_w = box_h * aspect
    x1 = int(cx - box_w / 2); y1 = int(cy - box_h / 2)
    x2 = int(cx + box_w / 2); y2 = int(cy + box_h / 2)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(rotated.shape[1], x2); y2 = min(rotated.shape[0], y2)
    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, out_size)


def detection(find_name):
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        return "無法開啟鏡頭"

    print("按 q 可隨時結束。開始辨識...")
    found = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)

        name = "No face"
        label = "No face"
        if results:
            face = max(results, key=lambda r: r["confidence"] * r["box"][2] * r["box"][3])
            x, y, w, h = face["box"]; x, y = abs(x), abs(y)

            crop = align_face(rgb, face["keypoints"], (x, y, w, h))
            if crop is not None:
                arr = crop.astype("float32") / 255.0
                arr = np.expand_dims(arr, axis=0)
                pred = model.predict(arr, verbose=0)[0]
                idx = int(np.argmax(pred))
                conf = float(pred[idx])
                name = cls_list[idx] if conf >= THRESHOLD else "Unknown"
                label = f"{name} ({conf:.2f})"

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition - press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif name == find_name:
            found = True
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if found:
        messagebox.showinfo("辨識結果", f"找到 {find_name} 了！")
        return f"解鎖成功：{find_name}"
    else:
        return "未辨識到指定使用者"