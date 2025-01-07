# Custom dataset converter for Severstal dataset
# https://www.kaggle.com/c/severstal-steel-defect-detection#

import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str,
                    default='../data/Severstal', help='root dir for data')
parser.add_argument('--annotation_file', type=str,
                    default='train.csv', help='annotation file')
parser.add_argument('--output_dir', type=str,
                    default='../data/Severstal/preprocessed', help='output dir')
args = parser.parse_args()

annotation = pd.read_csv(args.annotation_file)
print(f"Annotation file loaded with {len(annotation.columns)} columns, and {len(annotation)} records")

if args.subset == 'train':
    train_images = glob(os.path.join(args.root_dir, 'train_images/*.jpg'))
    print(f"Total number of train images: {len(train_images)}")


def decode_pixel(encoded_pixel):
    pixels = encoded_pixel.split(" ")
    decoded_pixel = []
    for i in range(0, int(len(pixels) / 2)):
        [decoded_pixel.append(str(pixel)) for pixel in range(int(pixels[i*2]), int(pixels[i*2]) + int(pixels[i*2 + 1]) - 1)]
    return " ".join(decoded_pixel)   

def pixel2coords(pixel, height):
    coords = divmod(pixel, height)
    return coords #WxH

def row_processing(row):
    row['DecodedPixels'] = decode_pixel(row['EncodedPixels'])
    return row
print("Start unwrapping pixels...")
tqdm.pandas()
annotation['DecodedPixels'] = annotation.EncodedPixels.progress_apply(decode_pixel)
total = 0
if not train_images is None:
    print("Start creating segmentation mask for training set...")
    for index, row in tqdm(annotation.iterrows(), total=annotation.shape[0]):
        image = cv2.imread(os.path.join(args.root_dir, 'train_images', row['ImageId']))
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Fullsize image
        cv2.imwrite('sample.jpg', grayscale)
        mask = np.zeros(image.shape, dtype=np.uint8)
        for pixel in row['DecodedPixels'].split(" "):
            w, h = pixel2coords(int(pixel), image.shape[0])
            mask[h, w] = (255, 255, 255)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center = int(x + w / 2)
            if center + 128 > image.shape[1]:
                new_image = image[:, image.shape[1] - 128:]
                new_mask = mask[:, image.shape[1] - 128:]
            elif center - 128 < 0:
                new_image = image[:, 0:128]
                new_mask = mask[:, 0:128]
            else:
                new_image = image[:, center - 128:center + 128]
                new_mask = mask[:, center - 128:center + 128]
            np.savez(os.path.join(args.output_dir, 'train', f"{row['ImageId'].split('.')[0]}_{mask_count}"), image=new_image, label=new_mask)
            mask_count += 1
        total += 1
print(f"Created {total} segmentation masks")