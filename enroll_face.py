import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import time
import math
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import ImageFont, ImageDraw, Image

# 引導用戶做的姿勢：每個階段顯示一句提示，平均分配要收集的張數
POSE_PROMPTS = [
    "正面看著鏡頭",
    "慢慢往左轉頭",
    "慢慢往右轉頭",
    "稍微抬頭",
    "稍微低頭",
    "靠近鏡頭一點",
    "往後退一點",
    "輕微左右晃動 + 自然表情",
]

FACE_SIZE = (224, 224)   # 跟訓練用的 image_size 一致

def put_chinese_text(image, text, position, font_path, font_size, color):
    image_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image_PIL)
    draw.text(position, text, font=font, fill=color)
    image = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)
    return image

OUT_SIZE = (640, 480)   # (寬, 高)

def align_face(img_rgb, keypoints, box, out_size=OUT_SIZE, scale=2.2):
    """
    以臉為中心，往外框一個較大的範圍（含背景），臉維持置中，輸出 out_size。
    scale = 框出來的大小相對臉的倍數；越大含越多背景。1.0=只有臉，2.2≈含不少背景
    """
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


def main(name):
    out_dir1 = "database/train"  # 這裡也可以改成參數
    out_dir2 = "database/val"
    num_faces = 600  # 想收集的臉部張數
    camera_id = 0  # 預設使用第一個攝影機
    min_confidence = 0.95  # MTCNN 偵測信心
    
    save_dir1 = os.path.join(out_dir1, name)
    save_dir2 = os.path.join(out_dir2, name)
    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    detector = MTCNN()
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("無法開啟鏡頭")
        return

    per_pose = max(1, num_faces // len(POSE_PROMPTS))
    saved = 0
    pose_idx = 0
    last_save = 0.0
    print("按 q 可隨時結束。開始收集...")

    while saved < num_faces:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)

        prompt = POSE_PROMPTS[min(pose_idx, len(POSE_PROMPTS) - 1)]
        face_crop = None

        if results:
            # 取信心最高、面積最大的那張臉
            face = max(results, key=lambda r: r["confidence"] * r["box"][2] * r["box"][3])
            if face["confidence"] >= min_confidence:
                x, y, w, h = face["box"]
                x, y = abs(x), abs(y)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 每 0.1 秒最多存一張，避免連續幀太像（資料缺乏多樣性）
                now = time.time()
                if now - last_save > 0.1:
                    face_crop = align_face(rgb, face["keypoints"], (x, y, w, h))

        if face_crop is not None:
            # 根據 saved 的值決定存儲到哪個目錄
            save_dir = save_dir1 if saved < 500 else save_dir2
            path = os.path.join(save_dir, f"{saved:03d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
            saved += 1
            last_save = time.time()
            pose_idx = saved // per_pose

        # 畫面提示
        frame = put_chinese_text(frame, f"姿勢: {prompt}", (10, 30), "simhei.ttf", 20, (0, 200, 255))
        frame = put_chinese_text(frame, f"已收集: {saved}/{num_faces}", (10, 65), "simhei.ttf", 20, (0, 200, 255))
        cv2.imshow("Enroll - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n完成，共存下 {saved} 張人臉到：{save_dir1} 和 {save_dir2}")
