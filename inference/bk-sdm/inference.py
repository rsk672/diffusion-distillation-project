
import os
import argparse
import time
from utils.inference_pipeline import InferencePipeline
from utils.misc import change_img_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-base")    
    parser.add_argument("--save_dir", type=str, default="./generated/bk-sdm-base",
                        help="$save_dir/{im256, im512} are created for saving 256x256 and 512x512 images")
    parser.add_argument("--captions_file", type=str, default="captions.txt", help="Path to the file containing prompts")    
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=512)
    parser.add_argument("--batch_sz", type=int, default=1)

    args = parser.parse_args()
    return args

def load_prompts_from_file(file_path):
    """Load prompts from a text file, each line is one prompt."""
    with open(file_path, 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]
    return prompts

if __name__ == "__main__":
    args = parse_args()

    pipeline = InferencePipeline(weight_folder=args.model_id,
                                 seed=args.seed,
                                 device=args.device)
    pipeline.set_pipe_and_generator()

    save_dir_src = os.path.join(args.save_dir, f'im{args.img_sz}')
    os.makedirs(save_dir_src, exist_ok=True)
    save_dir_tgt = os.path.join(args.save_dir, f'im{args.img_resz}')
    os.makedirs(save_dir_tgt, exist_ok=True)

    prompts_list = load_prompts_from_file(args.captions_file)
    params_str = pipeline.get_sdm_params()

    t0 = time.perf_counter()
    for batch_start in range(0, len(prompts_list), args.batch_sz):
        batch_end = batch_start + args.batch_sz

        val_prompts = prompts_list[batch_start:batch_end]
        img_names = [f"image_{i}.png" for i in range(batch_start, batch_end)]

        imgs = pipeline.generate(prompt=val_prompts,
                                 n_steps=args.num_inference_steps,
                                 img_sz=args.img_sz)

        for i, (img, img_name, val_prompt) in enumerate(zip(imgs, img_names, val_prompts)):
            img.save(os.path.join(save_dir_src, img_name))
            img.close()
            print(f"{batch_start + i}/{len(prompts_list)} | {img_name} {val_prompt}")
        print(f"---{params_str}")

    pipeline.clear()

    change_img_size(save_dir_src, save_dir_tgt, args.img_resz)
    print(f"{(time.perf_counter() - t0):.2f} sec elapsed")
