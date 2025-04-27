
from tqdm import tqdm 
from PIL import Image
import numpy as np
import argparse 
import torch 
import lmdb
import pandas as pd
import os 
import glob
import json

def store_arrays_to_lmdb(env, arrays_dict, start_index=0):
    """
    Store rows of multiple numpy arrays in a single LMDB.
    Each row is stored separately with a naming convention.
    """
    with env.begin(write=True) as txn:
        for array_name, array in arrays_dict.items():
            for i, row in enumerate(array):
                row_bytes = row.tobytes()
                data_key = f'{array_name}_{start_index+i}_data'.encode()
                txn.put(data_key, row_bytes)

def get_array_shape_from_lmdb(lmdb_path, array_name):
    with lmdb.open(lmdb_path) as env:
        with env.begin() as txn:
            image_shape = txn.get(f"{array_name}_shape".encode()).decode()
            image_shape = tuple(map(int, image_shape.split()))
    return image_shape 

# Example usage
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to CIFAR-10 images")
    parser.add_argument("--label_file", type=str, required=True, help="CSV file with labels")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB")
    parser.add_argument("--mapping_path", type=str, required=True, help="Path to save label-index mapping JSON")

    args = parser.parse_args()

    total_array_size = 10000000000  # Adjust to your need, set to 10GB by default for CIFAR-10
    env = lmdb.open(args.lmdb_path, map_size=total_array_size * 2) 

    # Read labels and create mapping
    labels_df = pd.read_csv(args.label_file)
    
    # Create a label-to-index mapping
    unique_labels = labels_df['label'].unique()
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    # Save the label-to-index mapping to a JSON file
    with open(args.mapping_path, 'w') as f:
        json.dump(label_to_index, f)

    labels_dict = dict(zip(labels_df['id'].astype(str), labels_df['label'].map(label_to_index)))

    image_list = []
    label_list = []

    image_files = sorted(glob.glob(os.path.join(args.data_path, "*.png")))

    for image_file in tqdm(image_files):
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        image = np.array(Image.open(image_file))
        image = image.transpose(2, 0, 1)  # Change to CHW format
        image_list.append(image)

        label = labels_dict[image_id]
        label_list.append(label)

    image_list = np.stack(image_list, axis=0)
    label_list = np.array(label_list)

    data_dict = {
        'images': image_list,
        'labels': label_list
    }

    # Store arrays in LMDB
    store_arrays_to_lmdb(env, data_dict)

    with env.begin(write=True) as txn:
        for key, val in data_dict.items():
            array_shape = np.array(val.shape)

            shape_key = f"{key}_shape".encode()
            shape_str = " ".join(map(str, array_shape))
            txn.put(shape_key, shape_str.encode())

if __name__ == "__main__":
    main()
