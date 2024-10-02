from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Đường dẫn đến thư mục chứa ảnh khuôn mặt đã xử lý
train_data_dir = 'processed_faces'  # Thay đổi đường dẫn nếu cần

# Thiết lập ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Tạo generator cho dữ liệu huấn luyện
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(160, 160),  # Kích thước đầu vào cho mô hình
    batch_size=32,
    class_mode='categorical'  # Sử dụng categorical cho bài toán phân loại
)

# Xây dựng mô hình
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(train_generator.num_classes, activation='softmax'))  # Số lớp đầu ra

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_generator, steps_per_epoch=train_generator.samples // train_generator.batch_size, epochs=10)

# Lưu mô hình
model.save('face_recognition_model.h5')  # Lưu mô hình
