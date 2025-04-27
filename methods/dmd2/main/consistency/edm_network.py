import torch
from main.consistency.cm.unet import UNetModel

def get_cifar10_edm_config():
    return dict(
        image_size=32,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=3,
        attention_resolutions=(2, ), # image_size=32 // attn=16
        dropout=0.1,
        channel_mult=(1, 2, 2, 1),
        # conv_resample=True,
        # dims=2,
        num_classes=10,
        use_checkpoint=False,
        use_fp16=False, #maybe will need to be true
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=32,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )

def get_edm_network():
    # if args.dataset_name == "imagenet":
    #     unet = EDMPrecond(
    #         img_resolution=args.resolution,
    #         img_channels=3,
    #         label_dim=args.label_dim,
    #         use_fp16=args.use_fp16,
    #         sigma_min=0,
    #         sigma_max=float("inf"),
    #         sigma_data=args.sigma_data,
    #         model_type="DhariwalUNet",
    #         **get_imagenet_edm_config()
    #     )
    # else:
    #     raise NotImplementedError
    
    # cfg = get_cifar10_edm_config()
    
    # model = UNetModel(**cfg)
    
    unet = UNetModel(
        **get_cifar10_edm_config()
    )
    
    return unet 


def get_scalings(sigma, sigma_data):
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
    c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
    return c_skip, c_out, c_in


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def sample_onestep(
    model,
    x,
    class_labels,
    sigmas,
    sigma_data,
    return_bottleneck=False
):
    """Single-step generation from a distilled model."""
    
    c_skip, c_out, c_in = [append_dims(s, x.ndim) for s in get_scalings(sigmas, sigma_data)]
    
    s_in = x.new_ones([x.shape[0]])
    sigmas_for_sampling = sigmas[0] * s_in

    rescaled_t = 1000 * 0.25 * torch.log(sigmas_for_sampling + 1e-44)
    model_output = model(c_in * x, rescaled_t.squeeze(), class_labels)
    
    if return_bottleneck:
        return model_output

    denoised = c_out * model_output + c_skip * x
    return denoised