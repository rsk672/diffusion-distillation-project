from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from utils import SDTextDataset
from accelerate.utils import set_seed
from accelerate import Accelerator
from tqdm import tqdm 
import numpy as np 
import argparse 
import logging 
import torch 
import glob 
import time 
import os 
from PIL import Image

logger = get_logger(__name__, log_level="INFO")

def create_generator(checkpoint_path, base_model=None):
    if base_model is None:
        generator = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet"
        ).float()
        generator.requires_grad_(False)
    else:
        generator = base_model

    counter = 0
    while True:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            break 
        except:
            print(f"fail to load checkpoint {checkpoint_path}")
            time.sleep(1)

            counter += 1 

            if counter > 100:
                return None

    print(generator.load_state_dict(state_dict, strict=True))
    return generator 

def get_x0_from_noise(sample, model_output, timestep):
    alpha_prod_t = (torch.ones_like(timestep).float() * 0.0047).reshape(-1, 1, 1, 1) 
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample

@torch.no_grad()
def sample(accelerator, current_model, vae, text_encoder, dataloader, args, teacher_pipeline=None):
    current_model.eval()
    all_images = [] 
    all_captions = [] 
    counter = 0 

    set_seed(args.seed+accelerator.process_index)
    
    logger.info(f"sampling...", main_process_only=True)

    for index, batch_prompts in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process, total=args.total_eval_samples // args.eval_batch_size // accelerator.num_processes):
        prompt_inputs = batch_prompts['text_input_ids_one'].to(accelerator.device).reshape(-1, batch_prompts['text_input_ids_one'].shape[-1])
        batch_text_caption_embedding = text_encoder(prompt_inputs)[0]

        timesteps = torch.ones(len(prompt_inputs), device=accelerator.device, dtype=torch.long)

        noise = torch.randn(len(prompt_inputs), 4, 
            args.latent_resolution, args.latent_resolution, 
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(index)
        ).to(accelerator.device) 

        eval_images = current_model(
            noise, timesteps.long() * (args.num_train_timesteps-1), batch_text_caption_embedding
        ).sample 

        eval_images = get_x0_from_noise(
            noise, eval_images, timesteps
        )

        eval_images = vae.decode(eval_images * 1 / 0.18215).sample.float()
        eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        eval_images = eval_images.contiguous() 

        gathered_images = accelerator.gather(eval_images)

        all_images.append(gathered_images.cpu().numpy())

        all_captions.append(batch_prompts['key'])

        counter += len(gathered_images)

        if counter >= args.total_eval_samples:
            break

    all_images = np.concatenate(all_images, axis=0)[:args.total_eval_samples] 
    if accelerator.is_main_process:
        print("all_images len ", len(all_images))

    all_captions = [caption for sublist in all_captions for caption in sublist]
    data_dict = {"all_images": all_images, "all_captions": all_captions}

    accelerator.wait_for_everyone()
    return data_dict 


def save_images_to_file(all_images, all_captions, save_dir_path):
    print(f'{all_images=} {all_captions[:len(all_images)]=}')
    for i, img in enumerate(all_images):
        img_name = f'{i}.png'
        image_pil = Image.fromarray(img) 
        image_pil.save(os.path.join(save_dir_path, img_name))
        image_pil.close()
    

@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to checkpoint")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--latent_resolution", type=int, default=64)
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--anno_path", type=str)
    parser.add_argument("--total_eval_samples", type=int, default=30000)
    parser.add_argument("--save_dir_path", type=str)
    args = parser.parse_args()

    accelerator_project_config = ProjectConfiguration(logging_dir='./')
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        project_config=accelerator_project_config
    )

    assert accelerator.num_processes == 1, "only support single gpu for now"

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 

    generator = None

    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        subfolder="vae"
    ).to(accelerator.device).float()

    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder"
    ).to(accelerator.device).float()
    
    logger.info(f"created vae and text_encoder", main_process_only=True)

    tokenizer = CLIPTokenizer.from_pretrained(
       "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
    )
    caption_dataset = SDTextDataset(args.anno_path, tokenizer, is_sdxl=False)

    caption_dataloader = torch.utils.data.DataLoader(
        caption_dataset, batch_size=args.eval_batch_size, 
        shuffle=False, drop_last=False, num_workers=8
    ) 
    caption_dataloader = accelerator.prepare(caption_dataloader)

    generator = create_generator(
        args.checkpoint_path, 
        base_model=generator
    )
    
    logger.info(f"created generator", main_process_only=True)

    generator = generator.to(accelerator.device)

    data_dict = sample(
        accelerator,
        generator,
        vae,
        text_encoder,
        caption_dataloader,
        args,
        teacher_pipeline=None
    )
  
    accelerator.wait_for_everyone()
    if args.save_dir_path:
        save_dir_path = args.save_dir_path
        print('Saving images...')
        save_images_to_file(data_dict['all_images'], data_dict['all_captions'], save_dir_path)

if __name__ == "__main__":
    evaluate()    