from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Tải mô hình đã huấn luyện
model = load_model('face_recognition_model.h5')

# Đường dẫn đến thư mục chứa ảnh kiểm tra
test_data_dir = r'D:\TAI LIEU\Tum_Tap_Code\Py\ML\Face_detection\dataset\test'

# Thiết lập ImageDataGenerator cho tập kiểm tra
test_datagen = ImageDataGenerator(rescale=1./255)

# Tạo generator cho dữ liệu kiểm tra
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Dự đoán cho toàn bộ tập kiểm tra
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)
predicted_labels = np.argmax(predictions, axis=-1)

# Nhãn thực tế
y_true = test_generator.classes

# Tính toán độ chính xác
accuracy = np.sum(predicted_labels == y_true) / len(y_true)
print(f'Do độ chính xác của mô hình: {accuracy * 100:.2f}%')

# Hiển thị một số dự đoán
for i in range(min(10, len(predicted_labels))):  # Hiển thị tối đa 10 dự đoán
    print(f'Ảnh: {test_generator.filenames[i]}, Dự đoán: {predicted_labels[i]}, Nhãn thực tế: {y_true[i]}')
