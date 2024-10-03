import cv2
import face_recognition
import pickle
import numpy as np

# Nạp known_face_encodings và known_face_names từ file mà bạn đã lưu trước đó
with open('face_data.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Khởi tạo webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Đọc khung hình từ webcam
    ret, frame = video_capture.read()

    # Chuyển đổi màu sắc của khung hình từ BGR (mặc định của OpenCV) sang RGB (mặc định của face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tìm vị trí và mã hóa khuôn mặt trong khung hình hiện tại từ webcam
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Duyệt qua từng khuôn mặt trong khung hình
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # So sánh với các khuôn mặt đã biết trong dataset
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Nếu tìm thấy khuôn mặt trùng khớp
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Tính khoảng cách (sai số) giữa các khuôn mặt
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            distance = round(face_distances[best_match_index], 2)

        # Vẽ hình chữ nhật xung quanh khuôn mặt và hiển thị tên
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, f"{name} ({distance})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Hiển thị khung hình
    cv2.imshow('Video', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam
video_capture.release()
cv2.destroyAllWindows()
