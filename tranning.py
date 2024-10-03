import face_recognition
import os
import pickle
# Đường dẫn đến folder dataset
DATASET_PATH = 'dataset/'

# Tạo danh sách lưu các đặc trưng khuôn mặt và tên
known_face_encodings = []
known_face_names = []

# Duyệt qua các thư mục của từng người trong dataset
for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person_name)
    
    # Duyệt qua từng ảnh trong thư mục của người đó
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        
        # Tải và xử lý ảnh
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        # Nếu ảnh có ít nhất một khuôn mặt
        if len(face_encodings) > 0:
            # Lấy đặc trưng khuôn mặt đầu tiên
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(person_name)

# Lưu known_face_encodings và known_face_names vào file
with open('face_data.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print(f"Đã nạp {len(known_face_encodings)} khuôn mặt từ dataset.")


