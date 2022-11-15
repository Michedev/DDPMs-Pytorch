import hydra
import pytorch_lightning as pl
import argparse

import torch
from omegaconf import OmegaConf
from path import Path
from tqdm import tqdm

from model.classifier_free_ddpm import GaussianDDPMClassifierFreeGuidance
import torchvision

from utils.paths import SCHEDULER

scheduler_paths = [p for p in SCHEDULER.files('*.yaml')]
scheduler_names = [x.basename().replace('.yaml', '') for x in scheduler_paths]
scheduler_map = {name: path for name, path in zip(scheduler_names, scheduler_paths)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', type=Path, required=True, help='Path to the checkpoint file')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device to use')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('-w', type=float, default=None, help='Class guidance')
    parser.add_argument('--scheduler', choices=scheduler_names, default=None,
                        help='use a custom scheduler', dest='scheduler')
    parser.add_argument('-T', type=int, default=None, help='Number of diffusion steps')
    return parser.parse_args()


@torch.no_grad()
def main():
    """
    Generate images from a trained model in the checkpoint folder
    """
    args = parse_args()

    print(args)
    run_path = args.run.abspath()
    pl.seed_everything(args.seed)
    assert run_path.exists(), run_path
    assert run_path.basename().endswith('.ckpt'), run_path
    print('loading model from', run_path)
    hparams = OmegaConf.load(run_path.parent / 'config.yaml')
    if args.T is not None:
        hparams.T = args.T
    if args.w is None:
        args.w = hparams.model.w
    model_hparams = hparams.model
    denoiser = hydra.utils.instantiate(model_hparams.denoiser_module)
    if args.scheduler is None:
        scheduler = hydra.utils.instantiate(hparams.scheduler)
    else:
        scheduler_conf = OmegaConf.load(scheduler_map[args.scheduler])
        scheduler_conf.T = hparams.noise_steps
        scheduler = hydra.utils.instantiate(scheduler_conf)
    model = GaussianDDPMClassifierFreeGuidance(
        denoiser_module=denoiser, T=model_hparams.T,
        w=args.w, p_uncond=model_hparams.p_uncond, width=model_hparams.width,
        height=model_hparams.height, input_channels=model_hparams.input_channels,
        num_classes=model_hparams.num_classes, logging_freq=1000, v=model_hparams.v,
        variance_scheduler=scheduler).to(args.device)
    model.load_state_dict(torch.load(run_path, map_location=args.device)['state_dict'])
    model = model.eval()
    images = []
    model.on_fit_start()

    for i_c in tqdm(range(model.num_classes)):
        c = torch.zeros((args.batch_size, model.num_classes), device=args.device)
        c[:, i_c] = 1
        gen_images = model.generate(batch_size=args.batch_size, c=c)
        images.append(gen_images)
    images = torch.cat(images, dim=0)
    # save images
    torchvision.utils.save_image(images, run_path.parent / 'generated_images.png', nrow=4, padding=2, normalize=True)


if __name__ == '__main__':
    main()
