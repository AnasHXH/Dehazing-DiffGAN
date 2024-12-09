import argparse
import logging
import os
import sys
import time
from collections import OrderedDict
import numpy as np
import torch
from IPython import embed
import lpips

import torchvision.utils as tvutils
from data import create_dataloader, create_dataset
from models import create_model

import utils as util

from utils import IRSDE  # Ensure these helper functions are in the same directory


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the dehazing model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output images.")
    parser.add_argument("--pretrained_g", type=str, required=True, help="Path to pretrained G model weights.")
    parser.add_argument("--pretrained_l", type=str, required=True, help="Path to pretrained L model weights.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Path to save logs.")
    parser.add_argument("--max_sigma", type=float, default=50, help="Max sigma value for SDE.")
    parser.add_argument("--T", type=int, default=100, help="Number of steps for SDE.")
    parser.add_argument("--schedule", type=str, default="cosine", help="Schedule for SDE (e.g., 'cosine').")
    parser.add_argument("--eps", type=float, default=0.005, help="Epsilon value for SDE.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output image filenames.")
    return parser.parse_args()


def setup_environment(output_dir, log_dir):
    """Setup directories, logger, and symbolic links."""
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize logger
    util.setup_logger(
        "base",
        log_dir,
        "test_log",
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")
    return logger

def test_model(
    input_dir,
    output_dir,
    pretrained_g,
    pretrained_l,
    max_sigma,
    T,
    schedule,
    eps,
    suffix,
    logger,
):
    """Run the model on the test dataset."""
    # Load dataset
    dataset_opt = {
        "name": "test_dataset",
        "mode": "LQ",
        "dataroot_LQ": input_dir,
        "phase": "test",
        "scale": 1,
        "data_type": "img",
        "LR_size": 128,  # Set appropriate value for LR image size
        "color": None,  # Add the 'color' key with a default value
    }
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(f"Number of test images in [{dataset_opt['name']}]: {len(test_set)}")

    # Load model
    model_opt = {
        "model": "latent_denoising",  # Add the model type
        "pretrain_model_G": pretrained_g,
        "pretrain_model_L": pretrained_l,
        "gpu_ids": [0],  # Add the GPU ID (use an empty list for CPU only)
        "is_train": False,  # Set to False since we are testing
        "dist": False,  # Set distributed training to False for testing
        "train": {},  # Add an empty 'train' key for testing
        "network_G": {  # Add the generator network configuration
            "which_model": "ConditionalNAFNet",
            "setting": {
                "img_channel": 8,
                "width": 64,
                "enc_blk_nums": [1, 1, 1, 28],
                "middle_blk_num": 1,
                "dec_blk_nums": [1, 1, 1, 1],
            },
        },
        "network_L": {  # Add the latent-level network configuration
            "which_model": "UNet",
            "setting": {
                "in_ch": 3,
                "out_ch": 3,
                "ch": 8,
                "ch_mult": [4, 8, 8, 16],
                "embed_dim": 8,
            },
        },
        "path": {  # Add paths for pretrained models and results
            "pretrain_model_G": pretrained_g,
            "pretrain_model_L": pretrained_l,
            "results_root": output_dir,
            "strict_load": True,  # Ensure strict loading of weights
        },
    }
    model = create_model(model_opt)
    device = model.device

    # Initialize SDE
    sde = IRSDE(
        max_sigma=max_sigma,
        T=T,
        schedule=schedule,
        eps=eps,
        device=device,
    )
    sde.set_model(model.model)

    # Test loop
    test_times = []
    for i, test_data in enumerate(test_loader):
        img_path = test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Encode and denoise
        LQ = test_data["LQ"]
        latent_LQ, hidden = model.encode(LQ.to(device))
        noisy_state = sde.noise_state(latent_LQ)

        model.feed_data(noisy_state, latent_LQ)
        tic = time.time()
        model.test(sde, hidden, save_states=False)
        toc = time.time()
        test_times.append(toc - tic)

        # Save results
        visuals = model.get_current_visuals(need_GT=False)
        SR_img = visuals["Output"][None, ...]
        output = util.tensor2img(SR_img.squeeze())  # uint8

        save_img_path = os.path.join(
            output_dir, img_name + suffix + ".png" if suffix else img_name + ".png"
        )
        util.save_img(output, save_img_path)

    logger.info(f"Average test time: {np.mean(test_times):.4f}s")



def main():
    """Main function."""
    args = parse_arguments()
    logger = setup_environment(args.output_dir, args.log_dir)
    logger.info("Starting the model testing...")

    test_model(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pretrained_g=args.pretrained_g,
        pretrained_l=args.pretrained_l,
        max_sigma=args.max_sigma,
        T=args.T,
        schedule=args.schedule,
        eps=args.eps,
        suffix=args.suffix,
        logger=logger,
    )


if __name__ == "__main__":
    main()
