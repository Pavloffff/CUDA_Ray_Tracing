import sys
from PIL import Image 
import struct 
import cv2 
import ctypes 
import numpy as np 
 
def parse_frame(path): 
    fin = open(path, 'rb') 
    (w, h) = struct.unpack('ii', fin.read(8)) 
    buff = ctypes.create_string_buffer(4 * w * h) 
    fin.readinto(buff) 
    fin.close() 
    img = Image.new('RGBA', (w, h)) 
    pix = img.load() 
    offset = 0 
    for j in range(h): 
        for i in range(w): 
            (r, g, b, a) = struct.unpack_from('cccc', buff, offset) 
            pix[i, j] = (ord(r), ord(g), ord(b), ord(a)) 
            offset += 4 
    return img 
 

frames = []
frame_count = int(sys.argv[1]) - 1

for i in range(frame_count):
    print("Converting", f"[{i+1}/{frame_count+1}]")
    img = parse_frame(f"frames/frame{i}.out")
    frames.append(img)

output_path = "result.gif"
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
print(output_path)
