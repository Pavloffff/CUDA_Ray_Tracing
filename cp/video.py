import sys
from PIL import Image
import struct
import ctypes
import cv2
import numpy as np

def parse_frame(path):
    """Читает бинарный файл кадра и возвращает изображение в формате NumPy"""
    with open(path, 'rb') as fin:
        (w, h) = struct.unpack('ii', fin.read(8))  # Читаем ширину и высоту
        buff = ctypes.create_string_buffer(4 * w * h)  # Буфер для RGBA данных
        fin.readinto(buff)
    
    img = Image.new('RGBA', (w, h))
    pix = img.load()
    offset = 0
    for j in range(h):
        for i in range(w):
            (r, g, b, a) = struct.unpack_from('cccc', buff, offset)
            pix[i, j] = (ord(r), ord(g), ord(b), ord(a))
            offset += 4
    
    # Конвертируем в OpenCV-совместимый формат (BGR)
    img = img.convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# Получаем количество кадров из аргументов командной строки
frame_count = int(sys.argv[1]) - 1

# Читаем первый кадр для получения размеров
first_frame = parse_frame(f"frames/frame0.out")
h, w, _ = first_frame.shape

# Создаем VideoWriter для записи MP4-видео
output_path = "result.mp4"
fps = 30  # Частота кадров (кадров в секунду)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# Обрабатываем кадры
for i in range(frame_count):
    print("Converting", f"[{i+1}/{frame_count}]")
    frame = parse_frame(f"frames/frame{i}.out")
    out.write(frame)

# Завершаем запись
out.release()
print(f"Видео сохранено как {output_path}")
