import os
import torch
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from deepface import DeepFace
from collections import Counter

# Load mô hình YOLO nhận diện khuôn mặt
face_model = YOLO("models/yolov8n-face.pt")
# emo_model = YOLO("models/emo-cls.pt")
# emo_model = YOLO("D:\DoAnCoSo\DACS\models\yolov8n-cls.pt")
# emo_model = YOLO("D:\DoAnCoSo\DACS\models\yolov8n(v2)-cls-best.pt")
emo_model = YOLO("D:/DoAnCoSo/DACS/Papers/LUẬN VĂN THẠC SĨ-20250507T015049Z-002/LUẬN VĂN THẠC SĨ/Train Results/Dataset cơ bản 15K (90_)/Yolov8(Ours)/yolov8_emotion(preprocess)/weights/best.pt")




face_recognizer = FaceAnalysis(name='buffalo_l', root='./models')
face_recognizer.prepare(ctx_id=0, det_size=(640, 640))


# Bản đồ cảm xúc
emotions_map = {
    "fear": "Positive",
    "sad": "Negative",
    "happy": "Positive",
    "neutral": "Neutral",
    "unknown": "Neutral"
}

identity_folder = "persons"


def classify_emotion(face):
    """Phân loại cảm xúc dựa vào mô hình DeepFace"""
    try:
        results = emo_model(face)
        if results:
            detected_emotion = results[0].names[results[0].probs.top1]
            return detected_emotion, emotions_map.get(detected_emotion, "Neutral")
    except:
        return "unknown", "Neutral"


def recognize_identity(face):
    """Nhận diện danh tính khuôn mặt từ cơ sở dữ liệu"""
    try:
        result = DeepFace.find(face, db_path=identity_folder, enforce_detection=False, model_name="ArcFace")
        if len(result[0]) > 0:
            return os.path.basename(result[0]['identity'][0])
        return "Unknown"
    except:
        return "Unknown"
