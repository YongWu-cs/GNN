from PIL import Image
import os
import shutil
from tqdm import tqdm

def is_black_and_white(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            if r != g != b:
                return False
    return True

def find_black_and_white_images(folder_path):
    black_and_white_images = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            file_path = os.path.join(folder_path, filename)
            if is_black_and_white(file_path):
                black_and_white_images.append(filename)
    return black_and_white_images

if __name__ == "__main__":
    folder_path = "Dataset/ColorImgs"
    black_and_white_images = find_black_and_white_images(folder_path)
    if black_and_white_images:
        for image_name in tqdm(black_and_white_images):
            target_folder = "Dataset/TestImgs"
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            src_path = os.path.join(folder_path, image_name)
            dst_path = os.path.join(target_folder, image_name)
            shutil.move(src_path, dst_path)

        print(f"黑白照片移动完成")
    else:
        print("未找到黑白图片。")