import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# def display_result(frames, emotion_counts, negative_faces):
#     """Hiển thị kết quả nhận diện khuôn mặt và cảm xúc"""
#     if not frames:
#         print("⚠ No frames to display.")
#         return  # Nếu không có frames hợp lệ, không làm gì cả
#
#     try:
#         if len(frames) == 4:
#             top_row = np.hstack((frames[0], frames[1]))
#             bottom_row = np.hstack((frames[2], frames[3]))
#             combined_frame = np.vstack((top_row, bottom_row))
#         else:
#             combined_frame = frames[0]
#
#         # Vẽ biểu đồ cảm xúc
#         plt.figure(figsize=(3, 3))
#         labels = list(emotion_counts.keys())
#         sizes = list(emotion_counts.values())
#         colors = ["red" if label == "Negative" else "green" if label == "Positive" else "gray" for label in labels]
#         plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
#         plt.title("Classroom Effectiveness")
#         plt.savefig("chart.png")
#
#         # Hiển thị biểu đồ
#         pie_chart = cv2.imread("chart.png")
#         pie_chart = cv2.resize(pie_chart, (500, combined_frame.shape[0]))
#
#         # Thêm cảnh báo
#         alert_text = "Alert: Increase engagement!" if emotion_counts.get("Negative", 0) > 0 else "Classroom is engaged!"
#         cv2.putText(pie_chart, alert_text, (50, pie_chart.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#
#         # Gộp khung hình chính và biểu đồ
#         result_display = np.hstack((combined_frame, pie_chart))
#
#         # Thêm khuôn mặt tiêu cực nếu có
#         if negative_faces:
#             neg_faces_panel = np.hstack(list(negative_faces))
#             neg_faces_panel = cv2.resize(neg_faces_panel, (result_display.shape[1], combined_frame.shape[0]))
#             result_display = np.vstack((result_display, neg_faces_panel))
#
#         # Hiển thị cửa sổ kết quả
#         cv2.imshow("Multi-Camera Face Recognition", result_display)
#
#     except Exception as e:
#         print(f"⚠ Error displaying results: {e}")


def display_result(frames, emotion_counts, negative_faces):
    """Hiển thị kết quả nhận diện khuôn mặt và cảm xúc"""
    if not frames:
        print("⚠ No frames to display.")
        return  # Nếu không có frames hợp lệ, không làm gì cả

    try:
        if len(frames) == 4:
            top_row = np.hstack((frames[0], frames[1]))
            bottom_row = np.hstack((frames[2], frames[3]))
            combined_frame = np.vstack((top_row, bottom_row))
        else:
            combined_frame = frames[0]

        # Vẽ biểu đồ cảm xúc
        plt.figure(figsize=(3, 3))
        labels = list(emotion_counts.keys())
        sizes = list(emotion_counts.values())

        # Ánh xạ màu sắc mới
        color_map = {
            "Negative": "red",     # Tiêu cực: Đỏ
            "Positive": "green",   # Tích cực: Xanh lá
            "Neutral": "gray",     # Bình thường: Xám
        }
        colors = [color_map.get(label, "gray") for label in labels]

        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.title("Classroom Effectiveness")
        plt.savefig("chart.png")

        # Hiển thị biểu đồ
        pie_chart = cv2.imread("chart.png")
        pie_chart = cv2.resize(pie_chart, (500, combined_frame.shape[0]))

        alert_text = "Alert: Increase engagement!" if emotion_counts.get("Negative", 0) > 0 else "Classroom is engaged!"
        cv2.putText(pie_chart, alert_text, (50, pie_chart.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        result_display = np.hstack((combined_frame, pie_chart))

        if negative_faces:
            neg_faces_panel = np.zeros_like(combined_frame)
            for i, face in enumerate(negative_faces):
                x, y, w, h = face
                cv2.rectangle(neg_faces_panel, (x, y), (x + w, y + h), (0, 0, 255), 2)
                time.sleep(2)

            neg_faces_panel = cv2.resize(neg_faces_panel, (result_display.shape[1], combined_frame.shape[0]))
            result_display = np.vstack((result_display, neg_faces_panel))

        cv2.imshow("Multi-Camera Face Recognition", result_display)

        cv2.imshow("Multi-Camera Face Recognition", result_display)

    except Exception as e:
        print(f"⚠ Error displaying results: {e}")