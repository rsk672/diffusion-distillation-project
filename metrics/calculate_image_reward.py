
import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_images_from_directory(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = Image.open(file_path).convert("RGB")
            images.append(np.array(image))
    return images

def compute_image_reward(images, captions, device):
    import ImageReward as RM
    model = RM.load("ImageReward-v1.0", device=device)
    rewards = []
    for image, prompt in tqdm(zip(images, captions), total=len(captions)):
        reward = model.score(prompt, Image.fromarray(image))
        rewards.append(reward)
    return np.mean(np.array(rewards))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate image rewards using ImageReward model.")
    parser.add_argument('--image-dir', type=str, required=True, help="Path to directory containing images")
    parser.add_argument('--captions-file', type=str, required=True, help="Path to file containing captions, one per line")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use for computation")

    args = parser.parse_args()

    images = load_images_from_directory(args.image_dir)

    with open(args.captions_file, 'r') as f:
        captions = [line.strip() for line in f]

    if len(images) != len(captions):
        raise ValueError("Number of images and captions must match.")

    average_reward = compute_image_reward(images, captions, device=args.device)
    
    print("Average Image Reward: {:.4f}".format(average_reward))