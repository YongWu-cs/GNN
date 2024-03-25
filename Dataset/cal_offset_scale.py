import os
from PIL import Image
import numpy as np
from skimage import color
from torchvision import transforms as T
from tqdm import tqdm

def compute_offsets_scales(dataset_directory):
    def load_images(files):
        for file in files:
            if file.endswith('.jpg'):
                im = Image.open(file)
                yield im

    L_accum, a_accum, b_accum = [], [], []
    num_files = 0
    for path in dataset_directory:
        files = [os.path.join(path, file) for file in os.listdir(path)]
        num_files += len(files)
        for im in tqdm(load_images(files), total=len(files)):
            im = im.resize((128, 128))
            im = np.array(im)
            if im.shape[2] != 3:
                im = np.stack([im[:, :, 0]] * 3, axis=2)
            im = color.rgb2lab(im).transpose((2, 0, 1))

            L_accum.append(im[0].flatten())
            a_accum.append(im[1].flatten())
            b_accum.append(im[2].flatten())

    L_flat = np.concatenate(L_accum)
    a_flat = np.concatenate(a_accum)
    b_flat = np.concatenate(b_accum)
    offsets = (np.mean(L_flat), np.mean(a_flat), np.mean(b_flat))
    scales = (np.std(L_flat), np.std(a_flat), np.std(b_flat))

    return offsets, scales, num_files 

dataset_directory = ['autodl-tmp/segment_dataset/train','autodl-tmp/segment_dataset/val','autodl-tmp/segment_dataset/test'] # Set this to your dataset directory
offsets, scales,_ = compute_offsets_scales(dataset_directory)

print("Offsets:", offsets)
print("Scales:", scales)
