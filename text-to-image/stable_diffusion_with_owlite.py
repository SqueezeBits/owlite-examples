"""Stable Diffusion Model with Owlite Integration

This script integrates the Stable Diffusion model with Owlite for model optimization and benchmarking.
"""

import argparse
import os

import numpy as np
import torch
from torch.backends import cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image

import owlite

parser = argparse.ArgumentParser(description="StableDiffusion with OwLite")

parser.add_argument(
    "--model-version",
    default="xl",
    dest="model_version",
    choices=["1.5", "xl"],
    type=str,
    help="Choose the Stable Diffusion model version for quantization. Options: 1.5 (smaller model) or xl (larger model, default)",
)
parser.add_argument(
    "--cache-dir",
    default=None,
    dest="cache_dir",
    type=str,
    help="Specify a directory to cache Hugging Face diffusion models for faster loading",
)
parser.add_argument(
    "--n-steps",
    type=int,
    dest="n_steps",
    default=30,
    help="Set the number of denoising steps for the diffusion model. Default is 30",
)
parser.add_argument(
    "--guidance",
    default=7.5,
    type=float,
    dest="guidance_scale",
    help="Adjust the classifier-free guidance scale. Must be greater than 1. Default is 7.5",
)
parser.add_argument(
    "--num-valid-images",
    default=1,
    dest="num_valid_images",
    type=int,
    help="Specify the number of validation images to use. Default is 1",
)
parser.add_argument(
    "--test-prompts",
    type=str,
    default="test_prompts.txt",
    dest="test_prompts",
    help="Provide a file containing test prompts for the model. Default is 'test_prompts.txt'",
)
parser.add_argument(
    "--negative-prompt",
    type=str,
    default=None,
    dest="negative_prompt",
    help="Negative prompt for the model. Default is None",
)
parser.add_argument(
    "--output-dir",
    default="./",
    type=str,
    dest="output_dir",
    help="Specify a directory to save PyTorch checkpoints and test images. Default is the current directory",
)
parser.add_argument(
    "--device",
    default=None,
    type=str,
    help="Choose a device for model execution (e.g., 'cuda', 'cpu')",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="Set a seed for initializing training",
)


##### OwLite #####
# Owlite arguments
owlite_parser = parser.add_subparsers(required=True).add_parser("owlite", help="Owlite arguments")
owlite_parser.add_argument(
    "--project",
    type=str,
    required=True,
    dest="owlite_project",
    help="Owlite project name",
)
owlite_parser.add_argument(
    "--baseline",
    type=str,
    required=True,
    dest="owlite_baseline",
    help="Owlite baseline name",
)
owlite_parser.add_argument(
    "--experiment",
    type=str,
    default=None,
    dest="owlite_experiment",
    help="Owlite experiment name",
)
owlite_parser.add_argument(
    "--duplicate-from",
    type=str,
    default=None,
    dest="owlite_duplicate_from",
    help="The name of Owlite experiment where the config to be duplicated is located",
)
owlite_parser.add_argument(
    "--calib-prompts",
    type=str,
    default="calib_prompts.txt",
    dest="owlite_calib_prompts",
    help="File with prompts for calibration",
)
owlite_parser.add_argument(
    "--ptq",
    action="store_true",
    dest="owlite_ptq",
    help="True if Owlite PTQ is applied",
)
owlite_parser.add_argument(
    "--download-engine",
    action="store_true",
    dest="download_engine",
    help="Download built TensorRT engine",
)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False

    # Load Stable Diffusion pipe
    if args.model_version == "1.5":
        args.model_path = "runwayml/stable-diffusion-v1-5"
    elif args.model_version == "xl":
        args.model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    if args.model_version != "xl":
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path, cache_dir=args.cache_dir, torch_dtype=torch.float32
        ).to(args.device)
        encoder_hidden_size = pipe.text_encoder.config.hidden_size
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_path, cache_dir=args.cache_dir, torch_dtype=torch.float32
        ).to(args.device)
        encoder_hidden_size = 2048

    unet = pipe.unet.eval()
    assert isinstance(unet, UNet2DConditionModel)

    # Initialize Owlite
    owl = owlite.init(
        project=args.owlite_project,
        baseline=args.owlite_baseline,
        experiment=args.owlite_experiment,
        duplicate_from=args.owlite_duplicate_from,
    )

    # Prepare arguments of unet
    batch_size = 2  # batch size is fixed at 2
    example_input_arg = (
        torch.cat(
            [torch.randn(1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size).to(args.device)]
            * batch_size
        ),  # latent_model_input
        torch.randint(low=0, high=1000, size=(), dtype=torch.float32).to(args.device),  # Timestamp
    )
    example_input_kwargs = {
        "encoder_hidden_states": torch.randn(
            batch_size, pipe.text_encoder.config.max_position_embeddings, encoder_hidden_size
        ).to(args.device),
        "timestep_cond": None,
        "cross_attention_kwargs": None,
        "added_cond_kwargs": None,
        "return_dict": False,
    }
    if isinstance(pipe, StableDiffusionXLPipeline):
        example_input_kwargs["added_cond_kwargs"] = {
            "text_embeds": torch.randn(batch_size, 1280).to(args.device)
            * torch.Tensor([0, 1]).reshape(2, 1).to(args.device),
            "time_ids": torch.Tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]] * 2).to(args.device),
        }

    # Convert Unet
    traced_unet = owl.convert(unet, *example_input_arg, **example_input_kwargs)
    traced_unet.config = unet.config
    traced_unet.device = unet.device
    traced_unet.dtype = unet.dtype
    if isinstance(pipe, StableDiffusionXLPipeline):
        traced_unet.add_embedding = unet.add_embedding
    pipe.unet = traced_unet

    # PTQ
    if args.owlite_ptq:
        with owlite.calibrate(pipe.unet):
            with open(args.owlite_calib_prompts) as file:
                calib_prompts = [line.rstrip("\n") for line in file]
            for prompt in calib_prompts:
                pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.n_steps,
                    guidance_scale=args.guidance_scale,
                )

    # export to onnx
    with torch.no_grad():
        owl.export(pipe.unet)

    # save model check point
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(pipe.unet.state_dict(), os.path.join(args.output_dir, "unet.ckpt"))

    # test model
    test(pipe, args.test_prompts, args)

    owl.benchmark(download_engine=args.download_engine)


def test(pipeline, prompts_path, args):
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    else:
        generator = None

    with open(prompts_path) as file:
        test_prompts = [line.rstrip("\n") for line in file]

    for idx, prompt in enumerate(test_prompts):
        images = []
        if generator is not None:
            generator.manual_seed(args.seed)
        for _ in range(args.num_valid_images):
            images.append(
                pipeline(
                    prompt=prompt,
                    num_inference_steps=args.n_steps,
                    negative_prompt=args.negative_prompt,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
            )
        img = np.concatenate(images, axis=1)
        img = Image.fromarray(img)
        img.save(os.path.join(args.output_dir, f"{idx}.png"))


if __name__ == "__main__":
    main()
