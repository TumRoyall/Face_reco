import os
import cv2
from mtcnn.mtcnn import MTCNN

# Khởi tạo MTCNN
detector = MTCNN()

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = 'dataset/'  # Thay đổi đường dẫn đến thư mục dữ liệu của bạn

output_dir = 'processed_faces/'

# Duyệt qua tất cả các thư mục và ảnh
for person in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person)
    if os.path.isdir(person_path):  # Kiểm tra xem có phải thư mục không
        # Tạo thư mục cho mỗi người trong thư mục output
        person_output_dir = os.path.join(output_dir, person)
        if not os.path.exists(person_output_dir):
            os.makedirs(person_output_dir)


        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            if img_name.endswith('.jpg') or img_name.endswith('.png'):  # Kiểm tra định dạng ảnh
                # Đọc ảnh
                image = cv2.imread(img_path)
                # Chuyển đổi màu ảnh từ BGR sang RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Phát hiện khuôn mặt
                faces = detector.detect_faces(image_rgb)

                for i, face in enumerate(faces):
                    x, y, width, height = face['box']
                    x, y = abs(x), abs(y)  # Đảm bảo giá trị x, y không âm
                    Face = image[y:y + height, x:x + width]

                  # Đưa kích thước về 160x160
                    face_resized = cv2.resize(Face, (160, 160))
                
                    # Lưu ảnh đã xử lý vào thư mục tương ứng với người
                    output_path = os.path.join(person_output_dir, f'processed_{img_name}')  # Thêm tên folder vào tên file
                    cv2.imwrite(output_path, face_resized)