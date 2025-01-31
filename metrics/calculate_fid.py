import argparse
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from cleanfid import fid

class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """ 
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__

def compute_fid(fake_dir, gt_dir, device,
    resize_size=None, feature_extractor="inception", 
    patch_fid=False):
    
    center_crop_trsf = CenterCropLongEdge()

    def resize_and_center_crop(image_np):
        image_pil = Image.fromarray(image_np)
        if patch_fid:
            if image_pil.size[0] >= 299 and image_pil.size[1] >= 299:
                image_pil = transforms.functional.center_crop(image_pil, 299)
        else:
            image_pil = center_crop_trsf(image_pil)

            if resize_size is not None:
                image_pil = image_pil.resize((resize_size, resize_size),
                                             Image.LANCZOS)
        return np.array(image_pil)

    if feature_extractor == "inception":
        model_name = "inception_v3"
    elif feature_extractor == "clip":
        model_name = "clip_vit_b_32"
    else:
        raise ValueError(
            "Unrecognized feature extractor [%s]" % feature_extractor)

    fid_score = fid.compute_fid(
        fdir1=fake_dir,
        fdir2=gt_dir,
        model_name=model_name,
        custom_image_tranform=resize_and_center_crop,
        use_dataparallel=False,
        device=device,
    )
    
    return fid_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID between sets of generated and real images.")
    parser.add_argument('--fake-dir', type=str, required=True, help="Path to directory containing generated images")
    parser.add_argument('--gt-dir', type=str, required=True, help="Path to directory containing ground truth images")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for computation (e.g. 'cuda:0' or 'cpu')")
    parser.add_argument('--resize-size', type=int, default=None, help="Size to resize images to, if needed")
    parser.add_argument('--feature-extractor', type=str, default='inception', choices=['inception', 'clip'], help="Feature extractor model")

    args = parser.parse_args()

    # Compute FID based on parsed arguments
    fid_score = compute_fid(
        fake_dir=args.fake_dir,
        gt_dir=args.gt_dir,
        device=args.device,
        resize_size=args.resize_size,
        feature_extractor=args.feature_extractor,
        patch_fid=False
    )

    print(f"FID Score: {fid_score}")