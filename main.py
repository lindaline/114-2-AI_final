import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import math
import numpy as np
import cv2
from mtcnn import MTCNN
from tensorflow.keras import models
 
MODEL_NAME = "resnet18"
MODEL_PATH = f"./{MODEL_NAME}.h5"
FOLDER_PATH = "database/train"   # 用子資料夾名稱當類別名
INPUT_SIZE = (224, 224)          # 跟訓練的 target_size 一致
THRESHOLD = 0.7                  # 低於此信心 → 判定為「未知」
CAMERA_ID = 0

model = models.load_model(MODEL_PATH)
# 類別名要照「訓練時的順序」= 資料夾排序後的順序（跟 flow_from_directory 一致）
cls_list = sorted(
    d for d in os.listdir(FOLDER_PATH) if os.path.isdir(os.path.join(FOLDER_PATH, d))
)
print("類別:", cls_list)
 
detector = MTCNN()
 
 
# def align_face(img_rgb, keypoints, box, out_size=INPUT_SIZE, margin=0.2):
#     """跟訓練時相同的對齊裁切，確保前處理一致"""
#     le, re = keypoints["left_eye"], keypoints["right_eye"]
#     angle = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
#     x, y, w, h = box
#     cx, cy = x + w / 2.0, y + h / 2.0
#     M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
#     rotated = cv2.warpAffine(img_rgb, M, (img_rgb.shape[1], img_rgb.shape[0]))
#     mx, my = int(w * margin), int(h * margin)
#     x1, y1 = max(0, int(x - mx)), max(0, int(y - my))
#     x2 = min(rotated.shape[1], int(x + w + mx))
#     y2 = min(rotated.shape[0], int(y + h + my))
#     face = rotated[y1:y2, x1:x2]
#     if face.size == 0:
#         return None
#     return cv2.resize(face, out_size)

def align_face(img_rgb, keypoints, box, out_size=INPUT_SIZE, scale=2.2):
    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0          # 臉的中心
    # 以臉中心旋轉拉正
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img_rgb, M, (img_rgb.shape[1], img_rgb.shape[0]))
    # 以臉中心，照 out_size 的比例往外抓一個大框（臉自然置中）
    aspect = out_size[0] / out_size[1]         # 寬/高 = 640/480
    box_h = h * scale
    box_w = box_h * aspect   # 讓框的比例跟 640:480 一致，不變形
    x1 = int(cx - box_w / 2); y1 = int(cy - box_h / 2)
    x2 = int(cx + box_w / 2); y2 = int(cy + box_h / 2)
    # 避免超出畫面邊界
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(rotated.shape[1], x2); y2 = min(rotated.shape[0], y2)
    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, out_size)

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("無法開啟鏡頭")
    raise SystemExit

print("按 q 可隨時結束。開始辨識...")
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    label = "No face"
    if results:
        face = max(results, key=lambda r: r["confidence"] * r["box"][2] * r["box"][3])
        x, y, w, h = face["box"]; x, y = abs(x), abs(y)
 
        crop = align_face(rgb, face["keypoints"], (x, y, w, h))  # RGB
        if crop is not None:
            arr = crop.astype("float32") / 255.0  # 跟訓練一致的歸一化
            arr = np.expand_dims(arr, axis=0)     # (1,224,224,3)
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
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()