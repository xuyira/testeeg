"""
从训练好的EEG扩散模型生成样本，保存为.npy格式
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_samples = []
    all_labels = []
    
    while len(all_samples) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        # ✅ 使用in_channels参数，默认为22（EEG通道数）
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        
        # ✅ 保持原始范围 [-1, 1]，不转换为图像格式
        # EEG数据不需要像图像那样转换为uint8
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        
        logger.log(f"created {len(all_samples) * args.batch_size} samples")

    # 合并所有样本
    arr = np.concatenate(all_samples, axis=0)
    arr = arr[: args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        
        # ✅ 保存为.npy格式（单个文件）
        out_path = os.path.join(logger.get_dir(), f"eeg_samples_{shape_str}.npy")
        logger.log(f"saving samples to {out_path}")
        logger.log(f"sample shape: {arr.shape}")  # (num_samples, 22, 64, 64)
        logger.log(f"sample range: [{arr.min():.4f}, {arr.max():.4f}]")
        np.save(out_path, arr)
        
        # 如果有标签，单独保存
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
            label_path = os.path.join(logger.get_dir(), f"eeg_labels_{args.num_samples}.npy")
            logger.log(f"saving labels to {label_path}")
            np.save(label_path, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        in_channels=22,  # ✅ EEG通道数
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

