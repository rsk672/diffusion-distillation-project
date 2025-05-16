import torch
import numpy as np
import dnnlib
import pickle
from tqdm import tqdm
import json
from accelerate import Accelerator
import scipy
import argparse

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
def calculate_inception_stats(all_images_tensor, evaluator, accelerator, evaluator_kwargs, feature_dim, batch_size):
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=accelerator.device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=accelerator.device)
    
    num_batches = ((len(all_images_tensor) - 1) // (batch_size * accelerator.num_processes) + 1) * accelerator.num_processes 
    all_batches = torch.arange(len(all_images_tensor)).tensor_split(num_batches)
    rank_batches = all_batches[accelerator.process_index::accelerator.num_processes]

    features = []
    for batch_indices in tqdm(rank_batches, disable=not accelerator.is_main_process):
        batch = all_images_tensor[batch_indices]
        f = evaluator(batch.permute(0, 3, 1, 2).to(accelerator.device), **evaluator_kwargs).to(torch.float64)
        features.append(f)

    features = torch.cat(features).cpu().numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def load_generated_images_from_npz(npz_path):
    npzfile = np.load(npz_path)
    images = npzfile['arr_0']
    images_tensor = torch.tensor(images, dtype=torch.uint8)
    return images_tensor

def calculate_fid_from_inception_stats(pred_mu, pred_sigma, ref_mu, ref_sigma):
    diff = pred_mu - ref_mu
    covmean, _ = scipy.linalg.sqrtm(pred_sigma @ ref_sigma, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff.T + np.trace(pred_sigma) + np.trace(ref_sigma) - 2 * np.trace(covmean)
    return fid

def evaluate(detector_url, stats_json_path, npz_path):
    accelerator = Accelerator()
    evaluator, evaluator_kwargs, feature_dim = create_evaluator(detector_url)
    ref_mu, ref_sigma = load_inception_stats_from_json(stats_json_path)
    images_tensor = load_generated_images_from_npz(npz_path)
    pred_mu, pred_sigma = calculate_inception_stats(images_tensor, evaluator, accelerator, evaluator_kwargs, feature_dim, batch_size=64)
    if accelerator.is_main_process:
        fid = calculate_fid_from_inception_stats(pred_mu, pred_sigma, ref_mu, ref_sigma)
        print(f"FID: {fid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inception', type=str, required=True, help="Path to inception network .pkl")
    parser.add_argument('--stats', type=str, required=True, help="Path to reference inception stats .json")
    parser.add_argument('--npz', type=str, required=True, help="Path to generated images .npz")
    args = parser.parse_args()
    evaluate(args.inception, args.stats, args.npz)
