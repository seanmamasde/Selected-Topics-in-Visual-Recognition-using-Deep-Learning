import os
import numpy as np
from PIL import Image

folder_path = './img'
output_npz = 'pred.npz'
images_dict = {}

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(folder_path, filename)
        image = Image.open(file_path).convert('RGB')
        img_array = np.array(image)
        img_array = np.transpose(img_array, (2, 0, 1))
        images_dict[filename] = img_array

np.savez(output_npz, **images_dict)
print(f"Saved {len(images_dict)} images to {output_npz}")
