from tqdm import tqdm 
import numpy as np 
import dnnlib
import pickle
import torch 
import scipy 
import json 


def load_inception_stats_from_json(json_path):
    with open(json_path, 'r') as f:
        ref_dict = json.load(f)
    
    mu = np.array(ref_dict['mu'])
    sigma = np.array(ref_dict['sigma'])
    
    return mu, sigma


def create_evaluator(detector_url):
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=False) as f:
        detector_net = pickle.load(f)

    detector_net.eval()
    return detector_net, detector_kwargs, feature_dim

@torch.no_grad()
def sample_for_fid(accelerator, current_model, args):
    timesteps = torch.ones(args.batch_size, device=accelerator.device, dtype=torch.long)
    current_model.eval()
    all_images = [] 
    all_images_tensor = []

    current_index = 0 

    all_labels = torch.arange(0, args.total_eval_samples, 
        device=accelerator.device, dtype=torch.long) % args.label_dim
    eye_matrix = torch.eye(args.label_dim, device=accelerator.device)

    while len(all_images_tensor) * args.batch_size * accelerator.num_processes < args.total_eval_samples:

        random_labels = all_labels[current_index:current_index+args.batch_size]
        one_hot_labels = eye_matrix[random_labels]
        noise = torch.randn(random_labels.shape[0], 3, 
            args.resolution, args.resolution, device=accelerator.device
        ) 

        current_index += args.batch_size
        eval_images = current_model(noise * args.conditioning_sigma, timesteps[:random_labels.shape[0]] * args.conditioning_sigma, one_hot_labels) 
        
        eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        eval_images = eval_images.contiguous() 

        gathered_images = accelerator.gather(eval_images)

        all_images.append(gathered_images.cpu().numpy())
        all_images_tensor.append(gathered_images.cpu())

    if accelerator.is_main_process:
        print("all_images len ", len(torch.cat(all_images_tensor, dim=0)))

    all_images = np.concatenate(all_images, axis=0)[:args.total_eval_samples]
    all_images_tensor = torch.cat(all_images_tensor, dim=0)[:args.total_eval_samples]

    accelerator.wait_for_everyone()
    return all_images_tensor 

@torch.no_grad()
def calculate_inception_stats(all_images_tensor, evaluator, accelerator, evaluator_kwargs, feature_dim, batch_size):
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=accelerator.device)
    sigma = torch.ones([feature_dim, feature_dim], dtype=torch.float64, device=accelerator.device)
    num_batches = ((len(all_images_tensor) - 1) // (batch_size * accelerator.num_processes ) + 1) * accelerator.num_processes 
    all_batches = torch.arange(len(all_images_tensor)).tensor_split(num_batches)
    rank_batches = all_batches[accelerator.process_index :: accelerator.num_processes]

    for i in tqdm(range(num_batches//accelerator.num_processes), unit='batch', disable=not accelerator.is_main_process):
        images = all_images_tensor[rank_batches[i]]
        features = evaluator(images.permute(0, 3, 1, 2).to(accelerator.device), **evaluator_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    mu = accelerator.reduce(mu) 
    sigma = accelerator.reduce(sigma)
    mu /= len(all_images_tensor)
    sigma -= mu.ger(mu) * len(all_images_tensor)
    sigma /= len(all_images_tensor) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))
