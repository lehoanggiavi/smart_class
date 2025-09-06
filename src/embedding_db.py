# build_multi_model_embeddings.py
import os
import pickle
from deepface import DeepFace


def build_multi_model_embedding_db(identity_folder, models=["SFace"]):
    for model_name in models:
        embedding_db = {}

        for person_name in os.listdir(identity_folder):
            person_path = os.path.join(identity_folder, person_name)
            if not os.path.isdir(person_path):
                continue

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name=model_name,
                        enforce_detection=False
                    )[0]["embedding"]

                    if person_name not in embedding_db:
                        embedding_db[person_name] = []

                    embedding_db[person_name].append(embedding)

                except Exception as e:
                    print(f"[ERROR] {img_path} ({model_name}): {e}")

        # Save DB
        with open(f'embeddings_{model_name}_1.pkl', 'wb') as f:
            pickle.dump(embedding_db, f)
        print(f"[INFO] Saved embeddings for {model_name}")

# Gọi hàm một lần để tạo toàn bộ embeddings
build_multi_model_embedding_db("D:/DoAnCoSo/DACS/persons")