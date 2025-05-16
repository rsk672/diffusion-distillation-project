
import argparse
import numpy as np
import torch
from accelerate import Accelerator
from lpips import LPIPS

def load_images_from_npz(npz_path):
    arr = np.load(npz_path)["arr_0"]
    images_tensor = torch.from_numpy(arr)
    if images_tensor.ndim == 3:
        images_tensor = images_tensor.unsqueeze(0)
    return images_tensor

@torch.no_grad()
def compute_diversity_score(lpips_loss_func, images, device):
    if isinstance(images, list):
        images = torch.stack([img if torch.is_tensor(img) else torch.from_numpy(img) for img in images])
    if images.ndim == 4 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    images = images.to(dtype=torch.float32, device=device)
    images = images / 255.0
    num_images = images.shape[0]
    loss_list = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            image1 = images[i].unsqueeze(0)
            image2 = images[j].unsqueeze(0)
            loss = lpips_loss_func(image1, image2)
            loss_list.append(loss.mean().item())
    return np.mean(loss_list) if loss_list else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True, help="Path to .npz with images")
    args = parser.parse_args()
    accelerator = Accelerator()
    lpips_loss_func = LPIPS(replace_pooling=True, reduction="none").to(accelerator.device)
    images = load_images_from_npz(args.npz_path)
    diversity_score = compute_diversity_score(lpips_loss_func, images, accelerator.device)
    if accelerator.is_main_process:
        print(f"Image Diversity Score: {diversity_score}")

if __name__ == "__main__":
    main()
