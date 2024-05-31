import numpy as np
import os
import sys
import glob
from PIL import Image
import argparse


def average_pooling(data, new_side_length):
    scale_factor = data.shape[0] / new_side_length
    pooled_data_shape = (new_side_length, new_side_length, data.shape[2] if len(data.shape) > 2 else 1)
    pooled_data = np.zeros(pooled_data_shape, dtype=data.dtype)
    
    for i in range(pooled_data_shape[0]):
        for j in range(pooled_data_shape[1]):
            if len(data.shape) > 2:  # For colored images
                for k in range(pooled_data_shape[2]):
                    pooled_data[i, j, k] = np.mean(
                        data[int(i * scale_factor):int((i + 1) * scale_factor),
                             int(j * scale_factor):int((j + 1) * scale_factor),
                             k])
            else:  # For grayscale images or 2D data
                pooled_data[i, j] = np.mean(
                    data[int(i * scale_factor):int((i + 1) * scale_factor),
                         int(j * scale_factor):int((j + 1) * scale_factor)])

    return pooled_data.squeeze()  # Remove single-dimensional entries

def read_raw(file_path, dimensions, dtype=np.uint8):
    data = np.fromfile(file_path, dtype=dtype)
    data = data.reshape(dimensions)  # Reshape according to the known dimensions
    return data

def process_file(file_path, target_resolutions, raw_dimensions=None, file_format_raw="uint8"):
    _, ext = os.path.splitext(file_path)
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    save_dir = os.path.join(os.path.dirname(file_path), file_name+"_downsampled")
    # add "do"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if ext in ['.png', '.jpg', '.jpeg']:
        with Image.open(file_path) as img:
            data = np.array(img)
    elif ext == '.npy':
        data = np.load(file_path)
    elif ext == '.raw':
        if raw_dimensions is None:
            print(f'Raw file dimensions not provided for {file_path}')
            return
        np_int = {
                "uint8": np.uint8,
                "uint16": np.uint16,
            }
        data = read_raw(file_path, raw_dimensions, dtype=np_int[file_format_raw])
    else:
        print(f'Unsupported file format: {ext}')
        return

    for res in target_resolutions:
        downsampled_data = average_pooling(data, res)
        file_name = file_name.split('_')[0]  # Remove resolution from file name (if present in the original file name)
        save_path = os.path.join(save_dir, f'{file_name}_{res}{ext}')
        if ext in ['.png', '.jpg', '.jpeg']:
            Image.fromarray(downsampled_data.astype('uint8')).save(save_path)
        elif ext == '.npy':
            np.save(save_path, downsampled_data)
        elif ext == '.raw':
            downsampled_data.tofile(save_path)
        print(f'Saved resolution {res} version to {save_path}')

def downsample_folder(folder_path, target_resolutions, raw_dimensions=None, file_format_raw="uint8"):
    supported_files = glob.glob(os.path.join(folder_path, '*.npy')) + \
                      glob.glob(os.path.join(folder_path, '*.png')) + \
                      glob.glob(os.path.join(folder_path, '*.jpg')) + \
                      glob.glob(os.path.join(folder_path, '*.raw'))

    for file_path in supported_files:
        process_file(file_path, target_resolutions, raw_dimensions, file_format_raw)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample images (jpg, png, npy) or raw files (uint8, uint16)')
    parser.add_argument('--data_dir', required=True, type=str, help='Directory containing the data files')
    parser.add_argument('--resolutions', required=True, type=str, help='Comma-separated list of target resolutions (e.g., 1024,512,256)')
    parser.add_argument('--raw_dims', type=str, help='Comma-separated dimensions for raw files (e.g., 512,512,512)')
    parser.add_argument('--raw_dtype', default='uint8', type=str, help='Data type for raw files (default: uint8)')

    args = parser.parse_args()

    folder_path = args.data_dir
    target_resolutions = list(map(int, args.resolutions.split(',')))

    raw_dimensions = None
    if args.raw_dims:
        raw_dimensions = tuple(map(int, args.raw_dims.split(',')))

    file_format_raw = args.raw_dtype

    downsample_folder(folder_path, target_resolutions, raw_dimensions, file_format_raw)
