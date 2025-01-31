import argparse
import os
from PIL import Image
from torchvision.transforms import CenterCrop
from torch.utils.data import DataLoader, Dataset
import torch
import clip as openai_clip
import numpy as np

class CenterCropLongEdge(object):
    def __call__(self, image):
        return CenterCrop(min(image.size))(image)

class CLIPScoreDataset(Dataset):
    def __init__(self, image_dir, captions, transform, preprocessor, how_many):
        super().__init__()
        self.images = [os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir))
                       if os.path.isfile(os.path.join(image_dir, fname))][:how_many]
        self.captions = captions[:how_many]
        self.transform = transform
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image_pil = self.transform(image)
        image_pil = self.preprocessor(image_pil)
        caption = self.captions[index]
        return image_pil, caption

@torch.no_grad()
def compute_clip_score(image_dir, captions, clip_model="ViT-B/32", device="cuda", how_many=30000):
    print("Computing CLIP score")
    
    if clip_model == "ViT-B/32":
        clip, clip_preprocessor = openai_clip.load("ViT-B/32", device=device)
        clip = clip.eval()
    elif clip_model == "ViT-G/14":
        import open_clip
        clip, _, clip_preprocessor = open_clip.create_model_and_transforms(
            "ViT-g-14", pretrained="laion2b_s12b_b42k")
        clip = clip.to(device)
        clip = clip.eval()
        clip = clip.float()
    else:
        raise NotImplementedError

    def resize_and_center_crop(image_np, resize_size=256):
        image_pil = image_np
        image_pil = CenterCropLongEdge()(image_pil)
        if resize_size is not None:
            image_pil = image_pil.resize((resize_size, resize_size), Image.LANCZOS)
        return image_pil

    def simple_collate(batch):
        images, captions = [], []
        for img, cap in batch:
            images.append(img)
            captions.append(cap)
        return images, captions

    dataset = CLIPScoreDataset(
        image_dir=image_dir, captions=captions,
        transform=resize_and_center_crop,
        preprocessor=clip_preprocessor,
        how_many=how_many
    )
    dataloader = DataLoader(
        dataset, batch_size=64,
        shuffle=False, num_workers=8,
        collate_fn=simple_collate
    )

    cos_sims = []
    count = 0
    for i, (imgs_pil, txts) in enumerate(dataloader):
        imgs = torch.stack(imgs_pil, dim=0).to(device)
        tokens = openai_clip.tokenize(txts, truncate=True).to(device)
        
        prepend_text = "A photo depicts "
        prepend_text_token = openai_clip.tokenize(prepend_text)[:, 1:4].to(device)
        prepend_text_tokens = prepend_text_token.expand(tokens.shape[0], -1)
        
        start_tokens = tokens[:, :1]
        new_text_tokens = torch.cat(
            [start_tokens, prepend_text_tokens, tokens[:, 1:]], dim=1)[:, :77]
        last_cols = new_text_tokens[:, 77 - 1:77]
        last_cols[last_cols > 0] = 49407  # eot token
        new_text_tokens = torch.cat([new_text_tokens[:, :76], last_cols], dim=1)
        
        img_embs = clip.encode_image(imgs)
        text_embs = clip.encode_text(new_text_tokens)

        similarities = torch.nn.functional.cosine_similarity(img_embs, text_embs, dim=1)
        cos_sims.append(similarities)
        count += similarities.shape[0]
        if count >= how_many:
            break
    
    clip_score = torch.cat(cos_sims, dim=0)[:how_many].mean()
    clip_score = clip_score.detach().cpu().numpy()
    return clip_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CLIP score for images and captions.")
    parser.add_argument('--image-dir', type=str, required=True, help="Directory containing images")
    parser.add_argument('--captions-file', type=str, required=True, help="Path to file containing captions, one per line")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use for computation")
    parser.add_argument('--clip-model', type=str, default='ViT-B/32', choices=['ViT-B/32', 'ViT-G/14'], help="CLIP model to use")
    parser.add_argument('--how-many', type=int, default=30000, help="Number of samples to compute")

    args = parser.parse_args()

    with open(args.captions_file, 'r') as f:
        captions = [line.strip() for line in f]
        
    clip_score = compute_clip_score(
        image_dir=args.image_dir,
        captions=captions,
        clip_model=args.clip_model,
        device=args.device,
        how_many=args.how_many
    )
    
    print(f"CLIP Score: {clip_score}")
