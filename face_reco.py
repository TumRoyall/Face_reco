import cv2
import numpy as np
import os
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

# Tải mô hình đã lưu
model = load_model('face_recognition_model.h5')

# Khởi tạo MTCNN
detector = MTCNN()

# Đường dẫn đến thư mục chứa ảnh
train_data_dir = 'processed_faces/'  # Thay đổi đường dẫn nếu cần

# Lấy tên của các folder tương ứng với ID
class_names = os.listdir(train_data_dir)

# Hàm để nhận diện khuôn mặt
def recognize_face(image):
    # Phát hiện khuôn mặt trong ảnh
    faces = detector.detect_faces(image)
    
    if len(faces) == 0:
        print("Không phát hiện khuôn mặt.")
        return None
    
    # Lặp qua tất cả các khuôn mặt được phát hiện
    for face in faces:
        # Lấy tọa độ khuôn mặt
        x, y, width, height = face['box']
        
        # Cắt khuôn mặt ra khỏi ảnh gốc
        face_image = image[y:y+height, x:x+width]
        
        # Chuyển đổi kích thước khuôn mặt về (160, 160)
        face_image = cv2.resize(face_image, (160, 160))
        
        # Chuyển đổi kiểu dữ liệu và chuẩn hóa
        face_image = np.array(face_image) / 255.0  # Chuẩn hóa
        face_image = np.expand_dims(face_image, axis=0)  # Thêm chiều cho batch size

        # Dự đoán lớp
        predictions = model.predict(face_image)
        class_index = np.argmax(predictions)

        # Lấy tên ID từ tên folder
        person_id = class_names[class_index]

        # Vẽ hộp bao quanh khuôn mặt và chỉ số lớp lên ảnh
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image, f'ID: {person_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Ví dụ về cách sử dụng hàm
if __name__ == "__main__":
    # Đọc ảnh từ file hoặc camera
    image = cv2.imread('test/im1.jpg')  # Thay thế bằng đường dẫn đến ảnh

    # Gọi hàm nhận diện
    result_image = recognize_face(image)

    # Hiển thị ảnh với các khuôn mặt đã được nhận diện
    cv2.imshow('Face Recognition', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
