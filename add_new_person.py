import cv2
import face_recognition
import os
import pickle
import time

# Nạp dữ liệu khuôn mặt đã biết từ file
try:
    with open('face_data.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
except FileNotFoundError:
    # Nếu file không tồn tại, khởi tạo danh sách trống
    known_face_encodings = []
    known_face_names = []

# Hàm thêm người mới và lưu ảnh
def add_new_person(name, dataset_path="dataset"):
    # Khởi tạo webcam
    video_capture = cv2.VideoCapture(0)
    
    # Tạo đường dẫn folder mới cho người mới trong dataset
    person_folder = os.path.join(dataset_path, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)  # Tạo folder nếu chưa có
        print(f"Đã tạo thư mục: {person_folder}")
    
    print(f"Đang chụp ảnh cho {name}...")

    image_count = 0  # Đếm số ảnh chụp
    start_time = time.time()  # Lưu thời gian bắt đầu

    while image_count < 5:  # Chụp 5 ảnh tự động
        ret, frame = video_capture.read()

        # Hiển thị video trên cửa sổ
        cv2.imshow('Video', frame)

        # Chuyển đổi ảnh từ BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tìm vị trí và mã hóa khuôn mặt
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings and time.time() - start_time >= 2:  # Chụp mỗi 2 giây nếu tìm thấy khuôn mặt
            # Thêm mã hóa khuôn mặt và tên người mới vào danh sách
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)

            # Lưu ảnh chụp vào folder
            image_count += 1
            image_path = os.path.join(person_folder, f"{name}_{image_count}.jpg")
            cv2.imwrite(image_path, frame)  # Lưu ảnh
            print(f"Đã lưu ảnh {image_path}")
            
            # Reset thời gian bắt đầu cho lần chụp tiếp theo
            start_time = time.time()

        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lưu encoding và tên vào file
    with open('face_data.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    # Giải phóng webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Gọi hàm để thêm người mới
name = input("Nhập tên của người mới: ")
add_new_person(name)
