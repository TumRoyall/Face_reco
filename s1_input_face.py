import cv2
import os
import time
import tkinter as tk
from tkinter import messagebox
from threading import Thread

# Hàm chụp ảnh
def capture_images(user_name):
    output_dir = f'dataset/{user_name}'

    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Khởi tạo camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở camera")
        return

    count = 0  # Đếm số ảnh đã chụp
    max_images = 5  # Số lượng ảnh cần chụp
    capture_interval = 0.2

    print(f"Đang chụp ảnh cho {user_name}...")

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không thể đọc khung hình")
            break

        # Hiển thị ảnh
        cv2.imshow('Camera', frame)

        # Lưu ảnh
        img_name = os.path.join(output_dir, f'img{count + 1}.jpg')
        cv2.imwrite(img_name, frame)
        print(f"Đã chụp ảnh: {img_name}")
        count += 1

        # Chờ trong khoảng thời gian xác định trước
        time.sleep(capture_interval)

    # Giải phóng camera và đóng tất cả các cửa sổ
    cap.release()
    cv2.destroyAllWindows()
    print("Đã chụp đủ số ảnh yêu cầu.")
    messagebox.showinfo("Thông báo", f"Đã chụp đủ {max_images} ảnh cho {user_name}.")

# Hàm để bắt đầu chụp ảnh khi nhấn nút
def start_capture():
    user_name = name_entry.get()
    if user_name:
        # Chạy hàm chụp ảnh trong một luồng mới để không làm tê liệt giao diện
        thread = Thread(target=capture_images, args=(user_name,))
        thread.start()
    else:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập tên của bạn.")

# Tạo giao diện người dùng
root = tk.Tk()
root.title("Chụp Ảnh Khuôn Mặt")

# Nhãn và ô nhập tên
name_label = tk.Label(root, text="Nhập tên của bạn:")
name_label.pack(pady=10)
name_entry = tk.Entry(root)
name_entry.pack(pady=10)

# Nút để bắt đầu chụp ảnh
start_button = tk.Button(root, text="Bắt đầu chụp ảnh", command=start_capture)
start_button.pack(pady=20)

# Chạy giao diện
root.mainloop()
