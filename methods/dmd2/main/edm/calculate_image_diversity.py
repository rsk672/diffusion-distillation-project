import torch
import numpy as np
from PIL import Image


@torch.no_grad()
def sample_images_for_diveristy(generator, num_images_per_class, device, args):
    """Generate images for each class using the generator model."""
    generator.eval()
    all_images_tensor = []
    eye_matrix = torch.eye(args.label_dim, device=device)

    for class_idx in range(args.label_dim):
        labels = torch.full((num_images_per_class,), class_idx, dtype=torch.long, device=device)
        one_hot_labels = eye_matrix[labels]
        
        noise = torch.randn(num_images_per_class, 3, args.resolution, args.resolution, device=device)
        timesteps = torch.ones(num_images_per_class, device=device, dtype=torch.long)  # Dummy timesteps
        
        eval_images = generator(noise * args.conditioning_sigma, timesteps * args.conditioning_sigma, one_hot_labels) 
        eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        
        all_images_tensor.append(eval_images.cpu())

    return all_images_tensor

@torch.no_grad()
def compute_diversity_score(lpips_loss_func, images, device):
    """Compute diversity score based on LPIPS for a list of image tensors."""
    images = [Image.fromarray(image.numpy()) for image in images]
    images = [image.resize((512, 512), Image.LANCZOS) for image in images]
    images = np.stack([np.array(image) for image in images], axis=0)
    
    images = torch.tensor(images).to(device).float() / 255.0
    images = images.permute(0, 3, 1, 2)
    
    num_images = images.shape[0]
    loss_list = []
    
    for i in range(num_images):
        for j in range(i + 1, num_images):
            
            image1 = images[i].unsqueeze(0)
            image2 = images[j].unsqueeze(0)
            
            loss = lpips_loss_func(image1, image2)
            loss_list.append(loss.item())

    return np.mean(loss_list)

@torch.no_grad()
def calculate_image_diversity(accelerator, generator, lpips_loss_func, args, num_images_per_class=10):
    all_class_images_tensor = sample_images_for_diveristy(generator, num_images_per_class, accelerator.device, args)

    class_diversity_scores = []
    for i, class_images_tensor in enumerate(all_class_images_tensor):
        diversity_score = compute_diversity_score(lpips_loss_func, class_images_tensor, accelerator.device)
        class_diversity_scores.append(diversity_score)
        print(f'diverity for class {i=} {diversity_score=}')
        
    mean_diversity_score = np.mean(class_diversity_scores)

    return mean_diversity_score
