import cv2
import torch
import os
import numpy as np
from collections import Counter, deque
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
import pickle
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import time
from functools import lru_cache

# Kiểm tra GPU
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tắt cảnh báo deprecated từ TensorFlow
warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.get_logger().setLevel('ERROR')

# Khởi tạo mô hình YOLO
face_model = YOLO(r"D:\Smart_Class\smart_class\MySystem\Face_Detect\yolov8n-face.pt")
emo_model = YOLO(r"D:\Smart_Class\smart_class\MySystem\Emo_Classifier\Yolov8(Ours)\yolov8_emotion(preprocess)\weights\best.pt")


# Cấu hình thư mục
identity_folder = r"D:\Smart_Class\smart_class\MySystem\Identity"
os.makedirs(identity_folder, exist_ok=True)
negative_faces_folder = r"D:\Smart_Class\smart_class\Negative_Faces"
os.makedirs(negative_faces_folder, exist_ok=True)

# Sử dụng webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Giảm độ phân giải
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

CONF_THRESHOLD = 0.5
MIN_FACE_SIZE = 50
BLUR_THRESHOLD = 100

emotions_map = {
    "fear": "Negative",
    "sad": "Negative",
    "happy": "Positive",
    "neutral": "Unknown",
    "unknown": "Unknown"
}

emotion_counts = Counter()
negative_faces = deque(maxlen=5)

model_thresholds = {
    "ArcFace": 0.5, 
    "VGG-Face":0.45,
    # "SFace": 0.6,
    "Facenet": 0.9
}

# Tải embedding DB
embedding_db = {}
for model_name in model_thresholds:
    with open(f'D:\Smart_Class\smart_class\Models\Embeddings\embeddings_{model_name}_1.pkl', 'rb') as f:
        embedding_db[model_name] = pickle.load(f)


# Cache nhận diện danh tính
@lru_cache(maxsize=100)
def recognize_identity(face_hash):
    face_crop = np.frombuffer(bytes.fromhex(face_hash), dtype=np.uint8)
    face_crop = cv2.imdecode(face_crop, cv2.IMREAD_COLOR)
    if not check_face_quality(face_crop):
        return "Unknown", 0.0
    votes = []
    for model_name, threshold in model_thresholds.items():
        try:
            target_embedding = DeepFace.represent(
                img_path=face_crop,
                model_name=model_name,
                enforce_detection=False
            )[0]["embedding"]
            min_dist = float('inf')
            best_match = "Unknown"
            for person, embeddings in embedding_db[model_name].items():
                for emb in embeddings:
                    dist = cosine(target_embedding, emb)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = person
            print(f"[{model_name}] Best match: {best_match} | Distance: {min_dist:.4f}")
            if min_dist <= threshold:
                votes.append(best_match)
            else:
                votes.append("Unknown")
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
            votes.append("Unknown")
    result = Counter(votes).most_common(1)[0]
    identity, count = result
    confidence = round(100 * count / len(model_thresholds), 2)
    if identity == "Unknown" or count < 1:
        return "Unknown", confidence
    return identity, confidence

def check_face_quality(face_crop):
    height, width = face_crop.shape[:2]
    if height < MIN_FACE_SIZE or width < MIN_FACE_SIZE:
        return False
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        return False
    return True

def classify_emotion(face):
    try:
        resized_face = cv2.resize(face, (224, 224))
        result = emo_model(resized_face, verbose=False)
        detected_emotion = result[0].names[result[0].probs.top1]
        confidence = result[0].probs.top1conf.item()
        general_emotion = emotions_map.get(detected_emotion, "Unknown")
        return detected_emotion, general_emotion, int(confidence * 100)
    except Exception as e:
        print(f"[ERROR] classify_emotion: {e}")
        return "unknown", "Unknown", 0

def get_pie_chart_image():
    fig, ax = plt.subplots(figsize=(3, 3))
    grouped_counts = {"Positive": 0, "Negative": 0, "Unknown": 0}
    for emotion, count in emotion_counts.items():
        group = emotions_map.get(emotion, "Unknown")
        grouped_counts[group] += count
    labels = list(grouped_counts.keys())
    sizes = list(grouped_counts.values())
    total = sum(sizes)
    if total == 0:
        ax.pie([1], labels=["No Data"], colors=["#808080"])
    else:
        colors = []
        for label in labels:
            if label == "Positive":
                colors.append("#00FF00")
            elif label == "Negative":
                colors.append("#FF0000")
            else:
                colors.append("#0000FF")
        ax.pie(sizes, labels=[f"{label}: {size/total*100:.1f}%" for label, size in zip(labels, sizes)],
               colors=colors, startangle=90)
    ax.set_title("Classroom Effectiveness")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    pie_chart = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return pie_chart

# Vòng lặp chính
frame_count = 0
last_pie_chart = None
skip_frames = 10
frame_skip_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_skip_counter += 1
    if frame_skip_counter % skip_frames != 0:
        continue  # Bỏ qua frame để tăng FPS

    # Giảm độ phân giải frame cho YOLO
    frame_small = cv2.resize(frame, (320, 240))
    results = face_model(frame_small, verbose=False)
    scale_x = frame.shape[1] / 320
    scale_y = frame.shape[0] / 240

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Chuyển tọa độ về kích thước gốc
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            face_crop = frame[y1:y2, x1:x2]

            # Phân loại cảm xúc
            detected_emotion, general_emotion, emo_conf = classify_emotion(face_crop)
            emotion_counts[detected_emotion] += 1
            if general_emotion == "Negative" and frame_count % 10 == 0:  # Lưu ít thường xuyên
                face_crop_resized = cv2.resize(face_crop, (100, 100))
                negative_faces.append(face_crop_resized)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{negative_faces_folder}/neg_{timestamp}.jpg", face_crop)

            # Nhận diện danh tính với cache
            _, buffer = cv2.imencode('.jpg', face_crop)
            face_hash = buffer.tobytes().hex()
            identity, identity_conf = recognize_identity(face_hash)

            # Chọn màu sắc
            color = (0, 0, 255) if general_emotion == "Negative" else (0, 255, 0) if general_emotion == "Positive" else (255, 0, 0)

            # Vẽ bounding box và nhãn
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{detected_emotion} ({emo_conf}%)", (x1, y1-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"ID: {identity} ({identity_conf}%)", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Cập nhật biểu đồ tròn
    frame_count += 1
    if frame_count % 10 == 0:  # Cập nhật mỗi 10 frame
        last_pie_chart = get_pie_chart_image()

    if last_pie_chart is not None:
        pie_chart = cv2.resize(last_pie_chart, (500, frame.shape[0]))
        negative_count = sum(emotion_counts[emotion] for emotion, category in emotions_map.items()
                            if category == "Negative" and emotion in emotion_counts)
        alert_text = "Alert: Increase engagement!" if negative_count > 0 else "Classroom is engaged!"
        cv2.putText(pie_chart, alert_text, (50, pie_chart.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        result_display = np.hstack((frame, pie_chart))

        if negative_faces:
            neg_faces_panel = np.hstack(list(negative_faces))
            target_width = result_display.shape[1]
            target_height = 200
            aspect_ratio = neg_faces_panel.shape[1] / neg_faces_panel.shape[0]
            new_height = int(target_width / aspect_ratio)
            if new_height > target_height:
                new_height = target_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = target_width
            neg_faces_panel = cv2.resize(neg_faces_panel, (new_width, new_height))
            panel = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            panel[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = neg_faces_panel
            result_display = np.vstack((result_display, panel))

        cv2.imshow("Face Recognition + Emotion Analysis", result_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()