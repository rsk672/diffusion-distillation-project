
import argparse
import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import piq

def load_images_from_directory(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = Image.open(file_path).convert("RGB")
            images.append(image)
    return images

def compute_diversity_score(lpips_loss_func, images, device):
    images = [image.resize((512, 512), Image.LANCZOS) for image in images]
    images = np.stack([np.array(image) for image in images], axis=0)
    images = torch.tensor(images).to(device).float() / 255.0
    images = images.permute(0, 3, 1, 2)

    num_images = images.shape[0]
    loss_list = []

    for i in tqdm(range(num_images)):
        for j in range(i + 1, num_images):
            image1 = images[i].unsqueeze(0)
            image2 = images[j].unsqueeze(0)
            loss = lpips_loss_func(image1, image2)
            loss_list.append(loss.item())
    return np.mean(loss_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute diversity score for images in a directory using LPIPS from PIQ.")
    parser.add_argument('--image-dir', type=str, required=True, help="Directory containing images")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use for computation")

    args = parser.parse_args()

    images = load_images_from_directory(args.image_dir)

    lpips_loss_func = piq.LPIPS(reduction='mean').to(args.device)

    diversity_score = compute_diversity_score(lpips_loss_func, images, args.device)

    print("Diversity Score: {:.4f}".format(diversity_score))
