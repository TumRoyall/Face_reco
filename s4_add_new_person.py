import cv2
import face_recognition
import os
import pickle

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
    
    print(f"Chụp ảnh cho {name}. Nhấn 's' để chụp, 'q' để thoát.")
    
    image_count = 0  # Đếm số ảnh chụp
    
    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)

        # Nếu nhấn 's' để chụp ảnh
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Lấy khung hình hiện tại
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Tìm vị trí và mã hóa khuôn mặt
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:
                # Thêm mã hóa khuôn mặt và tên người mới vào danh sách
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)
                print(f"Đã thêm {name} vào hệ thống.")
                
                # Lưu ảnh chụp vào folder
                image_count += 1
                image_path = os.path.join(person_folder, f"{name}_{image_count}.jpg")
                cv2.imwrite(image_path, frame)  # Lưu ảnh
                print(f"Đã lưu ảnh {image_path}")

                # Nếu đã chụp đủ số lượng ảnh mong muốn (ví dụ 5 ảnh), dừng lại
                if image_count >= 5:
                    print(f"Đã chụp đủ {image_count} ảnh cho {name}.")
                    break

            else:
                print("Không tìm thấy khuôn mặt, vui lòng thử lại.")
        
        # Nếu nhấn 'q' để thoát
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
